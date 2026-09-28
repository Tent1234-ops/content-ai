"""Read-only Phase 7 runner. No training, activation, Dataset import or provider fetch."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import re
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("ASR_LOCAL_FILES_ONLY", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from app.services.analysis_evaluation import (
    LABELS, REVIEW_FIELDS, audit_recommendation, classification_metrics, digest,
    human_review_summary, overlap_audit, scoring_exclusions, structural_summary,
)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def file_hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def new_output(path):
    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=False)
    return path


def inventory(args):
    out = new_output(args.out)
    unique = {}
    for path in sorted(Path(args.videos).resolve().rglob("*")):
        if path.suffix.lower() not in {".mp4", ".mov", ".mkv", ".webm"}:
            continue
        sha = file_hash(path)
        unique.setdefault(sha, []).append(str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path))
    cases = []
    for number, (sha, paths) in enumerate(unique.items(), 1):
        path = min(paths, key=len)
        cases.append({"case_id": f"existing-{number:02}", "video_path": path,
                      "media_sha256": sha, "duplicates": paths, "role": "regression",
                      "source_kind": "unverified", "expected_label": None,
                      "label_reviewed_by": "", "source_youtube_id": "", "source_channel_id": "",
                      "independence_confirmed_by": "", "provenance_note": "Previously available local clip; not a new blind test."})
    write(out / "cases.json", {"schema_version": 1, "cases": cases})
    write(out / "inventory.json", {"file_count": sum(map(len, unique.values())),
                                  "unique_media_count": len(cases), "missing": [
                                      "new_phone", "new_camera", "new_laptop", "self_recorded",
                                      "independently_labeled_out_of_scope"]})
    print(f"Inventory: {len(cases)} unique media files; labels and independent new clips still required.", flush=True)


def database_context(db):
    from app.database.models import SystemConfig
    from app.services.analysis_settings import capture_analysis_settings
    from app.services.classification import get_active_classification_model
    from app.services.classification_training import load_classification_artifact
    from app.services.dataset_eligibility import reference_transcript_rows
    from app.core.datetime_utils import utc_isoformat
    if db.query(SystemConfig).filter(SystemConfig.user_id.is_(None)).first() is None:
        raise ValueError("Configure analysis settings first; evaluation never creates production config")
    model = get_active_classification_model(db)
    if model is None:
        raise ValueError("No active model; evaluation will not activate or train one")
    artifact = load_classification_artifact(model.artifact_path)
    manifest_path = Path(artifact["dataset_manifest_path"])
    if artifact.get("dataset_manifest_sha256") and file_hash(manifest_path) != artifact["dataset_manifest_sha256"]:
        raise ValueError("Model dataset manifest hash mismatch")
    manifest = read(manifest_path)
    splits = {}
    for name, spec in manifest["artifacts"]["splits"].items():
        path = Path(spec["path"])
        if file_hash(path) != spec["sha256"]:
            raise ValueError(f"Frozen {name} split hash mismatch")
        splits[name] = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    reference_rows = []
    for row in reference_transcript_rows(db):
        data = {c.name: getattr(row, c.name) for c in row.__table__.columns}
        data = {key: utc_isoformat(value) if isinstance(value, datetime) else value for key, value in data.items()}
        reference_rows.append(data)
    pool = [{**r, "evaluation_pool": name} for name in ("train", "validation") for r in splits.get(name, [])]
    pool.extend({**r, "evaluation_pool": "reference"} for r in reference_rows)
    packages = {}
    for name in ("scikit-learn", "pythainlp", "faster-whisper", "ctranslate2"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "not_installed"
    context = {"at": datetime.now(timezone.utc).isoformat(), "settings": capture_analysis_settings(db),
               "python_version": sys.version, "packages": packages,
               "asr_language": os.getenv("ASR_LANGUAGE", "auto"),
               "artifact_path": model.artifact_path, "dataset_manifest_sha256": file_hash(manifest_path),
               "reference_sha256": digest(reference_rows), "reference_count": len(reference_rows),
               "code_sha256": {str(path.relative_to(ROOT)): file_hash(path) for path in [
                   ROOT / "app/services/recommendation.py", ROOT / "app/routes/analyze.py",
                   ROOT / "app/services/nlp.py", ROOT / "app/services/pipeline/core.py",
                   ROOT / "app/services/pipeline/domain_rules.py", ROOT / "models/speech_to_text.py",
                   ROOT / "app/services/analysis_evaluation.py"]}}
    return context, splits, reference_rows, pool


def review_template(cases):
    return [{"case_id": case["case_id"], "output_sha256": case["output_sha256"],
             "lane": check["lane"], "index": str(check["index"]), "keyword": check["keyword"],
             "reviewer": "", **{field: "" for field in REVIEW_FIELDS}, "notes": ""}
            for case in cases for check in case.get("recommendation_checks", [])]


def export_review(out, rows):
    with (out / "human-review.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case_id", "output_sha256", "lane", "index", "keyword", "reviewer", *REVIEW_FIELDS, "notes"])
        writer.writeheader()
        writer.writerows(rows)


def markdown_report(out, report):
    historical = report["historical_test"]["metrics"]
    coverage = report["coverage"]
    lines = ["# Phase 7 evaluation evidence", "", "## Status", "",
             "Independent end-to-end evaluation is INCOMPLETE until new labeled clips and human recommendation reviews are available.",
             "No model training, activation, Dataset import or production analysis save was performed.", "",
             f"New independent scored clips: {coverage['scored_new_clips']}",
             f"Missing new-clip classes: {', '.join(coverage['new_clip_labels_missing']) or 'none'}",
             f"Scored self-recorded clips: {coverage['self_recorded_scored']}", "",
             "## Historical classifier regression", "",
             "Previously evaluated frozen transcripts, NOT unseen videos and NOT an ASR test.",
             f"Samples: {historical['sample_size']}; accuracy: {historical['accuracy']}; macro F1: {historical['macro_f1']}",
             f"Unknown recall: {historical['unknown_recall']} (null means not evaluated)", "",
             "## Real local video runs", "",
             "These clips are regression cases unless independent provenance and gold labels are confirmed.", "",
             "| Case | Seconds | Predicted class | Confidence (not accuracy) | Suggestions / flagged | Result |",
             "| --- | ---: | --- | ---: | --- | --- |"]
    for case in report["cases"]:
        result = case.get("result", {})
        classification = result.get("recommendation", {}).get("classification", {})
        checks = case.get("recommendation_checks", [])
        lines.append(f"| {case['case_id']} | {case.get('duration_seconds', 0):.1f} | {case.get('predicted_label', 'error')} | "
                     f"{classification.get('confidence', 0):.4f} | {len(checks)} / {sum(bool(c['structural_issues']) for c in checks)} | "
                     f"[{case['case_id']}.json]({case['case_id']}.json) |")
    lines.extend(["", "## Recommendation evaluation", "",
                  "Automated checks validate repetition and Dataset evidence consistency. They reuse production term normalization and do not independently prove semantic relevance, transcript correctness or usefulness.",
                  "Human ratings are pending in human-review.csv. Suggested changes cannot be claimed to increase engagement without a separate outcome study.", "",
                  "```json", json.dumps(report["recommendation_structure"], indent=2), "```"])
    if "paired_reference_comparison" in report:
        lines.extend(["", "## Before / after reference recommendations", "",
                      "Same ASR, model, settings and reference fingerprint. Current trend ideas are excluded from the paired claim.",
                      "```json", json.dumps(report["paired_reference_comparison"], indent=2), "```"])
    lines.extend(["", "## Traceability", "", "- context.json: settings, model artifact hash, code hashes and reference fingerprint.",
                  "- reference-rows.json: local evidence snapshot, not new training data.",
                  "- report.json: every prediction, exclusion and structural check.",
                  "- human-review.csv: blank reviewer/rating fields until a person reviews the actual clip.", ""])
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run(args):
    from app.database.db import SessionLocal
    from app.routes.analyze import _build_recommendation
    from app.services.ai_pipeline import analyze_video
    from app.services.media_validation import validate_user_upload_duration
    from app.services.classification import classify_text_domain
    out = new_output(args.out)
    baseline = read(args.replay) if args.replay else None
    manifest = read(args.manifest) if args.manifest else None
    specs = [case["spec"] for case in baseline["cases"]] if baseline else manifest["cases"]
    ids = [case["case_id"] for case in specs]
    if len(ids) != len(set(ids)) or any(not re.fullmatch(r"[a-zA-Z0-9_-]+", case_id) for case_id in ids):
        raise ValueError("Case IDs must be unique safe filenames")
    db = SessionLocal()
    try:
        context, splits, references, pool = database_context(db)
        if baseline:
            for key in ("reference_sha256", "dataset_manifest_sha256"):
                if context[key] != baseline["context"][key]:
                    raise ValueError(f"Paired replay requires unchanged {key}; make a new baseline")
            for key in ("classification_model", "asr_model", "hook_duration_seconds", "upload_max_duration_seconds"):
                if context["settings"][key] != baseline["context"]["settings"][key]:
                    raise ValueError(f"Paired replay settings changed: {key}")
        write(out / "context.json", context)
        write(out / "reference-rows.json", references)
        historical = []
        for row in splits["test"]:
            overlap = overlap_audit(row, row["transcript"], pool)
            prediction = classify_text_domain(db, text=row["transcript"], require_active_model=True,
                                             model_snapshot=context["settings"]["classification_model"])
            historical.append({"dataset_id": row["dataset_id"], "expected_label": row["taxonomy_leaf_key"],
                               "predicted_label": prediction["taxonomy_leaf_key"], "confidence": prediction["confidence"],
                               "overlaps": overlap, "exclusions": ["training_or_reference_overlap"] if overlap else []})
        report = {"protocol_version": 1, "context": context,
                  "historical_test": {"label": "previously evaluated frozen transcript test; not new videos and not ASR evaluation",
                                      "metrics": classification_metrics(historical), "rows": historical},
                  "cases": [], "claims": {"engagement_improvement": "not_measured", "independent_recommendation_quality": "human_review_pending"}}
        seen_media = set()
        for spec in specs:
            entry = {"case_id": spec["case_id"], "spec": spec, "expected_label": spec.get("expected_label"),
                     "exclusions": ["analysis_not_completed"]}
            try:
                video = (ROOT / spec["video_path"]).resolve()
                media_hash = file_hash(video)
                if media_hash != spec["media_sha256"]:
                    raise ValueError("Media changed after test inventory")
                duration = validate_user_upload_duration(video, max_duration_seconds=context["settings"]["upload_max_duration_seconds"])
                print(f"Evaluating {spec['case_id']} ({duration:.1f}s)", flush=True)
                if baseline:
                    original = next(r for r in baseline["cases"] if r["case_id"] == spec["case_id"])
                    result = copy.deepcopy(original["result"])
                    result.pop("recommendation", None)
                else:
                    result = analyze_video(str(video), display_name="evaluation.mp4",
                                           hook_duration_seconds=context["settings"]["hook_duration_seconds"],
                                           asr_model_size=context["settings"]["asr_model"])
                result["analysis_settings"] = context["settings"]
                recommendation, _ = _build_recommendation(db, filename="evaluation.mp4", result=result,
                                                         settings_snapshot=context["settings"])
                result["recommendation"] = recommendation
                transcript = result.get("cleaned_transcript") or ""
                overlaps = overlap_audit(spec, transcript, pool)
                speech_ok = result.get("analysis", {}).get("stt_meta", {}).get("transcript_source") == "speech_to_text"
                entry.update(result=result, duration_seconds=duration, output_sha256=digest(result),
                             transcript_sha256=digest(transcript), overlaps=overlaps,
                             predicted_label=recommendation["domain"],
                             exclusions=scoring_exclusions(spec, overlaps, duplicate=media_hash in seen_media, speech_ok=speech_ok),
                             recommendation_checks=audit_recommendation(result, references))
                seen_media.add(media_hash)
            except Exception as exc:
                entry["error"] = str(exc)
            report["cases"].append(entry)
            write(out / f"{spec['case_id']}.json", entry)
            write(out / "report.json", report)
        report["new_clip_classification"] = classification_metrics(report["cases"])
        report["recommendation_structure"] = structural_summary(report["cases"])
        report["coverage"] = {"scored_new_clips": report["new_clip_classification"]["sample_size"],
                              "failed_case_count": sum(bool(r.get("error")) for r in report["cases"]),
                              "new_clip_labels_missing": [label for label in LABELS if not any(
                                  r.get("expected_label") == label and not r["exclusions"] for r in report["cases"])],
                              "self_recorded_scored": sum(r["spec"].get("source_kind") == "self_recorded" and not r["exclusions"] for r in report["cases"])}
        if baseline:
            report["paired_reference_comparison"] = {"baseline_sha256": file_hash(args.replay),
                "same_asr_transcripts": all(r.get("transcript_sha256") and r.get("transcript_sha256") == old.get("transcript_sha256")
                    for r, old in zip(report["cases"], baseline["cases"])),
                "before": baseline["recommendation_structure"], "after": report["recommendation_structure"],
                "trend_ideas_comparison": "excluded: collection time may differ"}
        write(out / "report.json", report)
        export_review(out, review_template(report["cases"]))
        markdown_report(out, report)
        print(json.dumps({"report": str(out / "report.json"), "coverage": report["coverage"],
                          "historical_test": report["historical_test"]["metrics"],
                          "structure": report["recommendation_structure"]}, ensure_ascii=True, indent=2), flush=True)
    finally:
        db.rollback()
        db.close()


def reviews(args):
    report = read(args.report)
    with Path(args.reviews).open(encoding="utf-8-sig", newline="") as handle:
        summary = human_review_summary(list(csv.DictReader(handle)), review_template(report["cases"]))
    print(json.dumps(summary, indent=2))


def summarize(args):
    path = Path(args.report).resolve()
    markdown_report(path.parent, read(path))
    print(str(path.parent / "report.md"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("inventory")
    prepare.add_argument("--videos", default="videos")
    prepare.add_argument("--out", required=True)
    evaluate = sub.add_parser("run")
    inputs = evaluate.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--manifest")
    inputs.add_argument("--replay", help="Previous report; reuse identical ASR for paired recommendation comparison")
    evaluate.add_argument("--out", required=True)
    review = sub.add_parser("reviews")
    review.add_argument("--report", required=True)
    review.add_argument("--reviews", required=True)
    summary = sub.add_parser("summarize")
    summary.add_argument("--report", required=True)
    args = parser.parse_args()
    {"inventory": inventory, "run": run, "reviews": reviews, "summarize": summarize}[args.command](args)


if __name__ == "__main__":
    main()
