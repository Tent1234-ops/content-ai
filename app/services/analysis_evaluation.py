"""Offline evaluation checks. Structural checks are not human usefulness scores."""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from types import SimpleNamespace


LABELS = ("phone", "camera", "laptop", "unknown")
REVIEW_FIELDS = ("relevant", "not_already_covered", "evidence_correct", "actionable")


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     default=str).encode("utf-8")).hexdigest()


def normalized_transcript(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text).casefold())


def diagnose_transcript_pair(asr_text: str, review: dict, classify) -> dict:
    """Compare unchanged ASR with an audio-verified transcript, never guessed corrections."""
    asr = classify(asr_text)
    result = {"asr_prediction": asr, "verified_transcript_prediction": None,
              "diagnosis": "awaiting_audio_verified_transcript", "is_new_test": False}
    if (not review.get("transcript_reviewed_by") or not review.get("listened_to_audio")
            or not str(review.get("verified_transcript") or "").strip()
            or review.get("expected_label") not in LABELS or not review.get("label_reviewed_by")):
        return result
    verified = classify(str(review["verified_transcript"]))
    gold = review["expected_label"]
    asr_label = asr.get("taxonomy_leaf_key")
    verified_label = verified.get("taxonomy_leaf_key")
    if asr.get("acceptance", {}).get("reason") == "scope_validation_unavailable":
        diagnosis = "scope_not_validated_cannot_attribute_error"
    elif verified_label != gold:
        diagnosis = "classification_error_remains_on_verified_text"
    elif asr_label != gold:
        diagnosis = "prediction_sensitive_to_transcription"
    else:
        diagnosis = "no_classification_error_in_this_pair"
    result.update(verified_transcript_prediction=verified, diagnosis=diagnosis,
                  transcript_changed=normalized_transcript(asr_text) != normalized_transcript(review["verified_transcript"]))
    return result


def overlap_audit(case: dict, transcript: str, pool: list[dict]) -> list[dict]:
    """Identity matches block scoring; long character overlaps require human review."""
    text = normalized_transcript(transcript)
    grams = {text[i:i + 12] for i in range(max(0, len(text) - 11))}
    matches = []
    for row in pool:
        reasons = []
        for key in ("source_youtube_id", "source_channel_id", "creator_group_key"):
            if case.get(key) and case[key] == row.get(key):
                reasons.append(key)
        other = normalized_transcript(str(row.get("transcript") or ""))
        if text and text == other:
            reasons.append("exact_transcript")
        elif len(text) >= 200 and len(other) >= 200:
            other_grams = {other[i:i + 12] for i in range(len(other) - 11)}
            containment = len(grams & other_grams) / max(1, min(len(grams), len(other_grams)))
            if containment >= 0.8:
                reasons.append("near_transcript_requires_review")
        if reasons:
            matches.append({"dataset_id": row.get("dataset_id"),
                            "pool": row.get("evaluation_pool"), "reasons": reasons})
    return matches


def scoring_exclusions(case: dict, overlaps: list[dict], *, duplicate: bool = False,
                       speech_ok: bool = True) -> list[str]:
    reasons = []
    if case.get("role") != "heldout":
        reasons.append("regression_not_new_holdout")
    if case.get("expected_label") not in LABELS or not case.get("label_reviewed_by"):
        reasons.append("human_label_missing")
    if not case.get("independence_confirmed_by"):
        reasons.append("independence_not_confirmed")
    if case.get("source_kind") == "self_recorded":
        if not case.get("provenance_note"):
            reasons.append("recording_provenance_missing")
    elif not case.get("source_youtube_id") or not case.get("source_channel_id"):
        reasons.append("source_identity_missing")
    if duplicate:
        reasons.append("duplicate_media")
    if not speech_ok:
        reasons.append("no_speech_transcript")
    if overlaps:
        reasons.append("training_or_reference_overlap")
    return reasons


def classification_metrics(rows: list[dict]) -> dict:
    """Macro F1 is over represented gold classes; abstention remains an error."""
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
    eligible = [r for r in rows if not r.get("exclusions") and r.get("expected_label") in LABELS]
    failed = [r for r in rows if r.get("error")]
    # A partial run must not look more accurate by silently dropping failed jobs.
    if failed:
        return {"sample_size": len(eligible), "accuracy": None, "macro_f1": None,
                "unknown_recall": None, "per_class": [], "confusion_matrix": None,
                "status": "incomplete_run", "failed_case_count": len(failed)}
    if not eligible:
        return {"sample_size": 0, "accuracy": None, "macro_f1": None,
                "unknown_recall": None, "per_class": [], "confusion_matrix": None}
    gold = [r["expected_label"] for r in eligible]
    predicted = [r["predicted_label"] for r in eligible]
    precision, recall, f1, support = precision_recall_fscore_support(
        gold, predicted, labels=list(LABELS), zero_division=0)
    represented = [i for i, count in enumerate(support) if count]
    unknown_index = LABELS.index("unknown")
    return {"sample_size": len(gold), "accuracy": float(accuracy_score(gold, predicted)),
            "macro_f1": float(sum(f1[i] for i in represented) / len(represented)),
            "macro_f1_labels": [LABELS[i] for i in represented],
            "unknown_recall": float(recall[unknown_index]) if support[unknown_index] else None,
            "unknown_prediction_count": predicted.count("unknown"),
            "per_class": [{"label": label, "support": int(support[i]),
                           "precision": float(precision[i]) if support[i] else None,
                           "recall": float(recall[i]) if support[i] else None,
                           "f1": float(f1[i]) if support[i] else None}
                          for i, label in enumerate(LABELS)],
            "confusion_labels": list(LABELS),
            "confusion_matrix": confusion_matrix(gold, predicted, labels=list(LABELS)).tolist()}


def audit_recommendation(result: dict, reference_rows: list[dict]) -> list[dict]:
    from app.services.recommendation import (
        _dataset_keyword_occurrences, _keyword_identity, recommendation_domain_for_taxonomy_leaf,
    )
    recommendation = result.get("recommendation", {})
    domain = recommendation.get("domain", "unknown")
    keyword_domain = recommendation_domain_for_taxonomy_leaf(domain)
    transcript = result.get("cleaned_transcript") or result.get("transcript") or ""
    observed = _dataset_keyword_occurrences(SimpleNamespace(transcript=transcript), domain=keyword_domain)
    normalized_text = normalized_transcript(transcript)
    refs = {int(r["dataset_id"]): r for r in reference_rows}
    occurrences = {}
    checks = []
    for lane in ("missing_keywords", "hook_keywords"):
        seen = set()
        for index, item in enumerate(recommendation.get(lane, [])):
            keyword = item["keyword"]
            identity = _keyword_identity(keyword, keyword_domain)
            reasons = []
            if identity in observed or normalized_transcript(keyword) in normalized_text:
                reasons.append("already_mentioned_or_synonym")
            if identity in seen:
                reasons.append("duplicate_suggestion")
            seen.add(identity)
            ids = item.get("supporting_dataset_row_ids") or []
            if not ids:
                reasons.append("no_dataset_evidence")
            if len(ids) != len(set(ids)) or item.get("support_count", 0) != len(set(ids)):
                reasons.append("invalid_support_count")
            frequency = 0
            for dataset_id in ids:
                row = refs.get(dataset_id)
                if row is None or row.get("taxonomy_leaf_key") != domain or row.get("data_split") != "train":
                    reasons.append("invalid_reference_row")
                    continue
                if dataset_id not in occurrences:
                    occurrences[dataset_id] = _dataset_keyword_occurrences(
                        SimpleNamespace(transcript=row["transcript"]), domain=keyword_domain)
                occurrence = occurrences[dataset_id].get(identity)
                if not occurrence:
                    reasons.append("term_not_found_in_cited_transcript")
                else:
                    frequency += occurrence["frequency"]
            if ids and frequency != item.get("total_frequency"):
                reasons.append("invalid_total_frequency")
            for example in item.get("supporting_examples", []):
                row = refs.get(example.get("dataset_id"))
                if not row or example.get("dataset_id") not in ids:
                    reasons.append("example_not_in_support")
                elif any(str(example.get(key) or "") != str(row.get(key) or "") for key in (
                        "video_url", "source_channel_id", "published_at", "statistics_captured_at")):
                    reasons.append("example_provenance_mismatch")
            if domain == "unknown":
                reasons.append("unknown_should_abstain")
            checks.append({"lane": lane, "index": index, "keyword": keyword,
                           "structural_issues": sorted(set(reasons)),
                           "human_review": "pending"})
    return checks


def structural_summary(cases: list[dict]) -> dict:
    checks = [check for case in cases for check in case.get("recommendation_checks", [])]
    return {"suggestion_count": len(checks),
            "suggestions_with_issues": sum(bool(c["structural_issues"]) for c in checks),
            "issues": dict(Counter(issue for c in checks for issue in c["structural_issues"])),
            "human_usefulness": "not_evaluated", "engagement_effect": "not_measured"}


def human_review_summary(reviews: list[dict], expected: list[dict]) -> dict:
    keys = {(r["case_id"], r["output_sha256"], r["lane"], r["index"]): r for r in expected}
    accepted = {}
    rejected = 0
    for review in reviews:
        key = tuple(str(review.get(k, "")) for k in ("case_id", "output_sha256", "lane", "index"))
        if key not in keys or key in accepted or not str(review.get("reviewer", "")).strip() or any(
                str(review.get(field, "")) not in {"0", "1"} for field in REVIEW_FIELDS):
            rejected += 1
            continue
        accepted[key] = review
    n = len(accepted)
    return {"reviewed": n, "reviewed_case_count": len({key[0] for key in accepted}),
            "pending": len(keys) - n, "rejected_rows": rejected,
            "scope": "reviewed suggestions only; includes regression clips, not a generalization estimate",
            "pass_rates": {field: sum(int(r[field]) for r in accepted.values()) / n if n else None
                           for field in REVIEW_FIELDS}, "engagement_effect": "not_measured"}
