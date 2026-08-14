from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from statistics import median
from typing import Dict, Iterable, List

from sqlalchemy.orm import Session

from app.database.models import DatasetContent, UserContent
from app.services.admin_settings import get_admin_config
from app.services.classification import classify_text_domain
from app.services.dataset_eligibility import production_transcript_query
from app.services.nlp import filter_tokens, normalize_text_for_nlp, run_nlp_pipeline, tokenize_text
from app.services.pipeline.core import (
    build_comparison_profile,
    build_dimension_status,
    extract_features,
    feature_to_keywords,
    normalize_asr_terms,
    normalize_space,
)
from app.services.pipeline.domain_rules import (
    DOMAIN_BASE,
    DOMAIN_DIMENSIONS_ORDER,
    DOMAIN_HINTS,
    detect_domain,
    infer_domain_from_features,
    normalize_domain,
)
from app.services.taxonomy import (
    UNKNOWN_LEAF_KEY,
    normalize_taxonomy_leaf,
    ready_leaf_keys,
)

DEFAULT_DURATION_BY_DOMAIN = {
    "audio": 75,
    "smartphone": 90,
    "food_drink": 45,
    "fashion": 60,
    "keyboard": 75,
    "mouse": 60,
    "skincare": 75,
    "general": 60,
    "phone": 90,
    "camera": 75,
    "laptop": 90,
    "headphone": 75,
    "hardware": 90,
    "general_food": 45,
    "drink": 45,
    "makeup": 60,
    "grooming": 60,
    "shirt": 60,
    "shoes": 60,
    "unknown": 60,
}

KEYWORD_DOMAIN_BY_TAXONOMY_LEAF = {
    "phone": "smartphone",
    "camera": "general",
    "laptop": "general",
    "audio": "audio",
    "headphone": "audio",
    "hardware": "general",
    "general_food": "food_drink",
    "drink": "food_drink",
    "makeup": "general",
    "grooming": "general",
    "shirt": "fashion",
    "shoes": "fashion",
}

GENERIC_RECOMMENDATION_BLACKLIST = {
    "video",
    "clip",
    "review",
    "youtube",
    "content",
    "subscribe",
    "follow",
    "like",
    "watch",
    "new",
    "best",
}

def _keyword_domain(leaf_key: str) -> str:
    return KEYWORD_DOMAIN_BY_TAXONOMY_LEAF.get(leaf_key, "general")


def _normalize_keyword(keyword: str) -> str:
    return (keyword or "").strip().lower()


def _keyword_is_domain_relevant(keyword: str, domain: str) -> bool:
    if not keyword:
        return False
    if domain == "general":
        return True
    lower = _normalize_keyword(keyword)
    hints = [*DOMAIN_HINTS.get(domain, []), *DOMAIN_BASE.get(domain, [])]
    hint_words: set[str] = set()
    for hint in hints:
        if not hint:
            continue
        normalized_hint = hint.lower()
        if normalized_hint in lower:
            return True
        for token in re.findall(r"[\u0E00-\u0E7Fa-z0-9]+", normalized_hint):
            if token:
                hint_words.add(token)
    keyword_tokens = set(re.findall(r"[\u0E00-\u0E7Fa-z0-9]+", lower))
    if len(keyword_tokens & hint_words) >= 2:
        return True
    return False


def _should_recommend_keyword(keyword: str, domain: str) -> bool:
    lower = _normalize_keyword(keyword)
    if not lower or lower in GENERIC_RECOMMENDATION_BLACKLIST:
        return False
    if domain != "general" and not _keyword_is_domain_relevant(lower, domain):
        return False
    return True


def _clean_profile_keywords(items: List[Dict[str, float]], domain: str, max_items: int) -> List[Dict[str, float]]:
    cleaned: List[Dict[str, float]] = []
    seen: set[str] = set()
    for item in items:
        keyword = str(item.get("keyword") or "").strip()
        if not keyword:
            continue
        lower = _normalize_keyword(keyword)
        if lower in seen:
            continue
        if not _should_recommend_keyword(lower, domain):
            continue
        seen.add(lower)
        cleaned.append({"keyword": keyword, "score": float(item.get("score", 0.0))})
        if len(cleaned) >= max_items:
            break
    return cleaned


def _weighted_increment(counter: Dict[str, float], key: str, weight: float) -> None:
    if not key:
        return
    counter[key] = counter.get(key, 0.0) + weight


def _fast_dataset_tokens(text: str) -> List[str]:
    normalized = normalize_space(text or "").lower()
    tokens = re.findall(r"[\u0E00-\u0E7Fa-z0-9][\u0E00-\u0E7Fa-z0-9\-]*", normalized)
    return filter_tokens(tokens)


def _fast_keyword_items(text: str, domain: str, max_keywords: int) -> List[Dict[str, float]]:
    normalized = normalize_space(text or "").lower()
    token_counts = Counter(_fast_dataset_tokens(normalized))
    scores: Dict[str, float] = {}

    for phrase in [*DOMAIN_BASE.get(domain, []), *DOMAIN_HINTS.get(domain, [])]:
        phrase_text = str(phrase or "").strip().lower()
        if not phrase_text:
            continue
        if phrase_text in normalized:
            scores[phrase_text] = max(scores.get(phrase_text, 0.0), 2.5 + len(phrase_text.split()) * 0.3)

    for token, count in token_counts.items():
        scores[token] = max(scores.get(token, 0.0), float(count))

    ranked = [
        {"keyword": keyword, "score": round(score, 3)}
        for keyword, score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    ]
    return ranked[:max_keywords]


def _analyze_dataset_profile_snapshot(
    text: str,
    title: str | None = None,
    category: str | None = None,
    max_keywords: int = 12,
    hook_duration: int = 60,
) -> Dict[str, object]:
    merged_text = " ".join(part for part in [title or "", text or "", category or ""] if part).strip()
    category_domain = normalize_domain(category)
    detected = detect_domain(f"{merged_text} {category or ''}")
    domain = category_domain if category_domain != "general" else detected
    if domain not in DEFAULT_DURATION_BY_DOMAIN:
        domain = detected if detected in DEFAULT_DURATION_BY_DOMAIN else "general"

    ranked_keywords = _fast_keyword_items(merged_text, domain, max_keywords)
    tokens = _fast_dataset_tokens(merged_text)
    hook_term_limit = min(12, max(4, int(hook_duration / 15)))
    return {
        "domain": domain,
        "nlp_result": {"top_keywords": ranked_keywords},
        "comparison_dimensions": [
            {"name": name, "confidence": 1.0}
            for name in DOMAIN_DIMENSIONS_ORDER.get(domain, [])[:8]
        ],
        "hook_terms": tokens[:hook_term_limit],
    }


def analyze_text_snapshot(
    text: str,
    title: str | None = None,
    max_keywords: int = 12,
    hook_duration: int = 60,
) -> Dict[str, object]:
    merged_text = " ".join(part for part in [title or "", text or ""] if part).strip()
    merged_text = normalize_asr_terms(merged_text)
    features = extract_features(merged_text)
    detected = detect_domain(merged_text)
    domain = infer_domain_from_features(features, detected)

    nlp_result = run_nlp_pipeline(merged_text, max_keywords=max_keywords)
    ranked_keywords = [
        {"keyword": item["keyword"], "score": float(item["score"]) }
        for item in nlp_result.get("top_keywords", [])
    ]
    existing_keywords = {item["keyword"] for item in ranked_keywords}
    for keyword in feature_to_keywords(features, domain):
        if keyword not in existing_keywords:
            ranked_keywords.append({"keyword": keyword, "score": 1.0})

    comparable_keywords, comparison_dimensions = build_comparison_profile(
        domain=domain,
        features=features,
        ranked_keywords=ranked_keywords,
        text=merged_text,
    )
    dimension_status = build_dimension_status(domain, comparison_dimensions)

    tokens = filter_tokens(tokenize_text(normalize_text_for_nlp(merged_text)))
    hook_term_limit = min(12, max(4, int(hook_duration / 15)))
    hook_terms = tokens[:hook_term_limit]
    return {
        "domain": domain,
        "features": features,
        "nlp_result": nlp_result,
        "comparable_keywords": comparable_keywords,
        "comparison_dimensions": comparison_dimensions,
        "dimension_status": dimension_status,
        "tokens": tokens,
        "hook_terms": hook_terms,
    }


def _duration_summary(durations: List[int], domain: str) -> Dict[str, object]:
    cleaned = sorted(duration for duration in durations if duration and duration > 0)
    if not cleaned:
        recommended = DEFAULT_DURATION_BY_DOMAIN.get(domain, 60)
        return {
            "recommended_seconds": recommended,
            "recommended_range": f"{max(15, recommended - 20)}-{recommended + 20} sec",
            "sample_size": 0,
            "source": "default",
        }

    if len(cleaned) < 3:
        default_value = DEFAULT_DURATION_BY_DOMAIN.get(domain, 60)
        mid = int(round((float(median(cleaned)) + default_value) / 2.0))
        return {
            "recommended_seconds": mid,
            "recommended_range": f"{max(15, mid - 25)}-{mid + 25} sec",
            "sample_size": len(cleaned),
            "source": "blended",
        }

    mid = int(round(float(median(cleaned))))
    low = max(10, int(round(cleaned[max(0, len(cleaned) // 4 - 1)])))
    high = int(round(cleaned[min(len(cleaned) - 1, (len(cleaned) * 3) // 4)]))
    if high < low:
        high = low
    return {
        "recommended_seconds": mid,
        "recommended_range": f"{low}-{high} sec",
        "sample_size": len(cleaned),
        "source": "dataset",
    }


def _source_label(
    source_prefix: str,
    sample_size: int,
    eligible_pool_size: int = 0,
) -> str:
    if sample_size <= 0:
        return "No eligible verified transcript samples found for this taxonomy leaf"
    if source_prefix == "youtube":
        pool_label = (
            f" from {eligible_pool_size} eligible same-category clips"
            if eligible_pool_size > sample_size
            else ""
        )
        return (
            "Human-reviewed YouTube Creative Commons transcripts "
            f"({sample_size} high-performing clips{pool_label})"
        )
    if source_prefix == "google":
        return f"Google Trends dataset ({sample_size} same-type items)"
    if source_prefix == "tiktok":
        return f"TikTok live/dataset trends ({sample_size} same-type items)"
    return f"{source_prefix} dataset ({sample_size} same-type samples)"


def _build_evidence(profile: Dict[str, object], *, source_prefix: str) -> Dict[str, object]:
    duration = profile.get("recommended_duration")
    if not isinstance(duration, dict):
        duration = {}
    sample_size = int(profile.get("sample_size") or 0)
    eligible_pool_size = int(profile.get("eligible_pool_size") or sample_size)
    duration_sample_size = int(duration.get("sample_size") or 0)
    duration_source = str(duration.get("source") or "default")
    return {
        "source": profile.get("source") or source_prefix,
        "dataset_sources": profile.get("dataset_sources") or [],
        "dataset_versions": profile.get("dataset_versions") or [],
        "data_source_label": _source_label(
            source_prefix,
            sample_size,
            eligible_pool_size,
        ),
        "dataset_sample_size": sample_size,
        "eligible_pool_size": eligible_pool_size,
        "source_platform_counts": profile.get("source_platform_counts") or {},
        "transcript_source_counts": profile.get("transcript_source_counts") or {},
        "language_counts": profile.get("language_counts") or {},
        "collection_strategy_counts": profile.get("collection_strategy_counts") or {},
        "selection_rule": profile.get("selection_rule") or "none",
        "license_name": profile.get("license_name") or "",
        "verification_status": profile.get("verification_status") or "",
        "duration_source": duration_source,
        "duration_sample_size": duration_sample_size,
        "duration_samples": profile.get("duration_samples") or [],
        "exemplar_titles": profile.get("exemplar_titles") or [],
        "dataset_row_ids": profile.get("dataset_row_ids") or [],
        "source_record_ids": profile.get("source_record_ids") or [],
        "keyword_score_explanation": (
            "Keyword gap uses only human-reviewed YouTube Creative Commons transcripts "
            "in the same taxonomy leaf. High-performing examples are selected from real "
            "YouTube statistics using average views per day and engagement rate captured "
            "during collection."
            if sample_size > 0
            else "No eligible same-type transcript samples were available; no dataset keyword evidence was generated."
        ),
        "duration_explanation": (
            f"Recommended duration uses {duration_sample_size} same-type duration samples from the {source_prefix} dataset."
            if duration_sample_size > 0
            else "Recommended duration falls back to the default range for this content type because no same-type duration samples were available."
        ),
    }


def build_dataset_profiles(
    db: Session,
    *,
    source_prefix: str = "youtube",
    limit: int = 150,
) -> List[Dict[str, object]]:
    admin_config = get_admin_config(db)
    if source_prefix == "youtube" and not admin_config.enable_youtube_trending:
        return []
    if source_prefix == "google" and not admin_config.enable_google_trends:
        return []
    profiles = [
        build_dataset_profile_for_domain(
            db,
            domain=leaf_key,
            source_prefix=source_prefix,
            limit=limit,
        )
        for leaf_key in sorted(ready_leaf_keys(db))
    ]
    return [profile for profile in profiles if int(profile["sample_size"]) > 0]


def _platform_key(source_platform: str | None, source_prefix: str) -> str:
    value = str(source_platform or source_prefix).strip().lower()
    for platform in ("youtube", "google", "tiktok"):
        if value.startswith(platform):
            return platform
    return value or source_prefix


def _high_performing_rows(
    rows: List[DatasetContent],
    *,
    limit: int,
) -> tuple[List[DatasetContent], Dict[int, float]]:
    if not rows:
        return [], {}
    ranked = sorted(
        rows,
        key=lambda row: (
            -float(row.trend_score or 0.0),
            -int(row.views or 0),
            int(row.dataset_id),
        ),
    )
    selection_size = min(limit, max(10, math.ceil(len(ranked) * 0.4)))
    selected = ranked[:selection_size]
    denominator = max(len(selected) - 1, 1)
    weights = {
        int(row.dataset_id): 1.0 + 2.0 * (1.0 - (index / denominator))
        for index, row in enumerate(selected)
    }
    return selected, weights


def build_dataset_profile_for_domain(
    db: Session,
    *,
    domain: str,
    source_prefix: str = "youtube",
    limit: int = 150,
) -> Dict[str, object]:
    """Build recommendation evidence from the exact canonical category rows used."""
    canonical_domain = normalize_taxonomy_leaf(domain)
    keyword_domain = _keyword_domain(canonical_domain)
    admin_config = get_admin_config(db)
    eligible_rows: List[DatasetContent] = []
    if canonical_domain in ready_leaf_keys(db):
        eligible_rows = (
            production_transcript_query(db, train_only=True)
            .filter(DatasetContent.is_keyword_recommendation_eligible.is_(True))
            .filter(DatasetContent.source_platform.like(f"{source_prefix}%"))
            .filter(DatasetContent.taxonomy_leaf_key == canonical_domain)
            .order_by(DatasetContent.trend_score.desc(), DatasetContent.dataset_id.asc())
            .limit(max(limit * 3, 150))
            .all()
        )
    rows, row_weights = _high_performing_rows(eligible_rows, limit=limit)

    keyword_scores: Dict[str, float] = {}
    dimension_scores: Dict[str, float] = {}
    hook_scores: Dict[str, float] = {}
    durations: List[int] = []
    platform_counts: Counter[str] = Counter()
    transcript_source_counts: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()
    collection_strategy_counts: Counter[str] = Counter()

    for row in rows:
        base_text = " ".join(
            part
            for part in [row.title or "", row.transcript or "", row.category or ""]
            if part
        ).strip()
        weight = row_weights.get(int(row.dataset_id), 1.0)
        for item in _fast_keyword_items(
            base_text,
            keyword_domain,
            admin_config.max_keywords_display,
        ):
            _weighted_increment(
                keyword_scores,
                str(item["keyword"]),
                weight * float(item["score"]),
            )
        for name in DOMAIN_DIMENSIONS_ORDER.get(keyword_domain, [])[:8]:
            if _keyword_is_domain_relevant(name, keyword_domain):
                confidence = 1.0 if name.lower() in base_text.lower() else 0.35
                _weighted_increment(dimension_scores, name, weight * confidence)
        hook_limit = min(12, max(4, int(admin_config.hook_analysis_duration / 15)))
        for term in _fast_dataset_tokens(base_text)[:hook_limit]:
            _weighted_increment(hook_scores, term, weight)
        if (
            row.is_duration_recommendation_eligible
            and row.duration_seconds
            and int(row.duration_seconds) > 0
        ):
            durations.append(int(row.duration_seconds))
        platform_counts[_platform_key(row.source_platform, source_prefix)] += 1
        transcript_source_counts[str(row.transcript_source or "unknown")] += 1
        language_counts[str(row.language or "und")] += 1
        collection_strategy_counts[
            str(row.collection_strategy or "classification_diverse")
        ] += 1

    raw_top_keywords = [
        {"keyword": keyword, "score": round(score, 3)}
        for keyword, score in sorted(
            keyword_scores.items(), key=lambda item: (-item[1], item[0])
        )[:20]
    ]
    top_keywords = _clean_profile_keywords(
        raw_top_keywords,
        keyword_domain,
        max_items=10,
    )
    top_dimensions = [
        {"keyword": name, "score": round(score, 3)}
        for name, score in sorted(
            dimension_scores.items(), key=lambda item: (-item[1], item[0])
        )[:8]
    ]
    raw_hook_keywords = [
        {"keyword": keyword, "score": round(score, 3)}
        for keyword, score in sorted(
            hook_scores.items(), key=lambda item: (-item[1], item[0])
        )[:20]
    ]
    hook_keywords = _clean_profile_keywords(
        raw_hook_keywords,
        keyword_domain,
        max_items=8,
    )

    sample_size = len(rows)
    if sum(platform_counts.values()) != sample_size:
        raise RuntimeError("Recommendation evidence platform counts do not match sample size")

    return {
        "domain": canonical_domain,
        "sample_size": sample_size,
        "eligible_pool_size": len(eligible_rows),
        "top_keywords": top_keywords,
        "top_dimensions": top_dimensions,
        "hook_keywords": hook_keywords,
        "recommended_duration": _duration_summary(durations, canonical_domain),
        "duration_samples": sorted(durations),
        "exemplar_titles": [str(row.title) for row in rows[:5]],
        "dataset_row_ids": [int(row.dataset_id) for row in rows],
        "source_record_ids": [str(row.source_record_id) for row in rows],
        "source": "youtube_cc_human_verified" if rows else "none",
        "dataset_sources": sorted({str(row.dataset_source) for row in rows}),
        "dataset_versions": sorted({str(row.dataset_version) for row in rows}),
        "source_platform_counts": dict(platform_counts),
        "transcript_source_counts": dict(transcript_source_counts),
        "language_counts": dict(language_counts),
        "collection_strategy_counts": dict(collection_strategy_counts),
        "selection_rule": (
            "top_40_percent_by_average_views_per_day_and_engagement_rate"
            if rows
            else "none"
        ),
        "license_name": "YouTube Creative Commons Attribution" if rows else "",
        "verification_status": "human_verified" if rows else "",
    }


def _find_profile(profiles: Iterable[Dict[str, object]], domain: str) -> Dict[str, object]:
    domain = normalize_taxonomy_leaf(domain)
    for profile in profiles:
        if profile["domain"] == domain:
            return profile
    return {
        "domain": domain,
        "sample_size": 0,
        "eligible_pool_size": 0,
        "top_keywords": [
            {"keyword": keyword, "score": 0.0}
            for keyword in DOMAIN_BASE.get(_keyword_domain(domain), [])[:8]
        ],
        "top_dimensions": [
            {"keyword": name, "score": 0.0}
            for name in DOMAIN_DIMENSIONS_ORDER.get(_keyword_domain(domain), [])[:8]
        ],
        "hook_keywords": [],
        "recommended_duration": _duration_summary([], domain),
        "exemplar_titles": [],
        "duration_samples": [],
        "dataset_row_ids": [],
        "source_record_ids": [],
        "source": "none",
        "dataset_sources": [],
        "dataset_versions": [],
        "source_platform_counts": {},
        "transcript_source_counts": {},
        "language_counts": {},
        "collection_strategy_counts": {},
        "selection_rule": "none",
        "license_name": "",
        "verification_status": "",
    }


def build_recommendation_from_text(
    db: Session,
    *,
    title: str | None,
    text: str,
    source_prefix: str = "youtube",
    profile_limit: int = 150,
) -> Dict[str, object]:
    admin_config = get_admin_config(db)
    user_snapshot = analyze_text_snapshot(
        text,
        title,
        max_keywords=admin_config.max_keywords_display,
        hook_duration=admin_config.hook_analysis_duration,
    )
    classification = classify_text_domain(
        db,
        title=title,
        text=text,
        source_prefix=source_prefix,
        profile_limit=profile_limit,
    )
    selected_domain = str(user_snapshot["domain"])
    if classification.get("is_unknown"):
        selected_domain = UNKNOWN_LEAF_KEY
    elif float(classification.get("confidence", 0.0)) >= 0.45:
        selected_domain = str(classification["domain"])
    user_keywords = [item["keyword"] for item in user_snapshot["nlp_result"].get("top_keywords", [])]
    user_keywords.extend(user_snapshot["comparable_keywords"])
    recommendation = build_recommendation_from_analysis_data(
        db,
        domain=selected_domain,
        user_keywords=user_keywords,
        dimension_status=user_snapshot["dimension_status"],
        hook_terms=user_snapshot["hook_terms"],
        source_prefix=source_prefix,
        profile_limit=profile_limit,
    )
    recommendation["classification"] = classification
    return recommendation


def build_recommendation_from_analysis_data(
    db: Session,
    *,
    domain: str,
    user_keywords: List[str],
    dimension_status: List[Dict[str, object]],
    hook_terms: List[str],
    source_prefix: str = "youtube",
    profile_limit: int = 150,
) -> Dict[str, object]:
    domain = normalize_taxonomy_leaf(domain)
    keyword_domain = _keyword_domain(domain)
    profile = build_dataset_profile_for_domain(
        db,
        domain=domain,
        source_prefix=source_prefix,
        limit=profile_limit,
    )

    normalized_user_keywords = {keyword.lower() for keyword in user_keywords}
    missing_keywords = []
    for item in profile["top_keywords"]:
        keyword = str(item["keyword"]).strip()
        if not keyword:
            continue
        if keyword.lower() in normalized_user_keywords:
            continue
        missing_keywords.append({"keyword": keyword, "score": round(float(item["score"]), 3)})
        if len(missing_keywords) >= 6:
            break

    seen_hook_terms: set[str] = set()
    hook_keywords = []
    for term in hook_terms:
        normalized_term = str(term or "").strip().lower()
        if not normalized_term or normalized_term in seen_hook_terms or normalized_term in normalized_user_keywords:
            continue
        if normalized_term in GENERIC_RECOMMENDATION_BLACKLIST:
            continue
        if keyword_domain != "general" and not _keyword_is_domain_relevant(normalized_term, keyword_domain):
            continue
        seen_hook_terms.add(normalized_term)
        hook_keywords.append({"keyword": term, "score": 0.0})
        if len(hook_keywords) >= 5:
            break

    if not hook_keywords:
        for item in profile["hook_keywords"]:
            keyword = str(item["keyword"]).strip()
            if not keyword:
                continue
            lower_keyword = keyword.lower()
            if lower_keyword in normalized_user_keywords or lower_keyword in seen_hook_terms:
                continue
            if lower_keyword in GENERIC_RECOMMENDATION_BLACKLIST:
                continue
            seen_hook_terms.add(lower_keyword)
            hook_keywords.append({"keyword": keyword, "score": round(float(item["score"]), 3)})
            if len(hook_keywords) >= 5:
                break

    user_dimensions = {row["name"]: row for row in dimension_status}
    missing_dimensions = []
    for item in profile["top_dimensions"]:
        name = item["keyword"]
        user_dimension = user_dimensions.get(name)
        if user_dimension and user_dimension["status"] == "present":
            continue
        missing_dimensions.append(
            {
                "name": name,
                "score": round(float(item["score"]), 3),
                "user_status": user_dimension["status"] if user_dimension else "missing",
            }
        )
        if len(missing_dimensions) >= 6:
            break

    deduped_user_keywords = []
    seen = set()
    for keyword in user_keywords:
        low = keyword.lower()
        if low in seen:
            continue
        seen.add(low)
        deduped_user_keywords.append(keyword)

    return {
        "domain": domain,
        "user_keywords": deduped_user_keywords[:12],
        "missing_keywords": missing_keywords,
        "hook_keywords": hook_keywords,
        "missing_dimensions": missing_dimensions,
        "recommended_duration": profile["recommended_duration"],
        "dataset_profile": profile,
        "evidence": _build_evidence(profile, source_prefix=source_prefix),
    }


def build_recommendation_from_saved_content(
    db: Session,
    *,
    content_id: int,
    source_prefix: str = "youtube",
    profile_limit: int = 150,
    user_id: int | None = None,
    allow_admin: bool = False,
) -> Dict[str, object] | None:
    query = db.query(UserContent).filter(UserContent.content_id == content_id)
    if user_id is not None and not allow_admin:
        query = query.filter(UserContent.user_id == user_id)

    content = query.first()
    if content is None:
        return None
    return build_recommendation_from_text(
        db,
        title=content.title,
        text=content.transcript or content.title,
        source_prefix=source_prefix,
        profile_limit=profile_limit,
    )


def compare_dataset_profiles(
    db: Session,
    *,
    left_source: str = "youtube",
    right_source: str = "google",
    limit: int = 150,
) -> Dict[str, object]:
    left_profiles = build_dataset_profiles(db, source_prefix=left_source, limit=limit)
    right_profiles = build_dataset_profiles(db, source_prefix=right_source, limit=limit)

    left_map = {profile["domain"]: profile for profile in left_profiles}
    right_map = {profile["domain"]: profile for profile in right_profiles}
    domains = sorted(set(left_map) | set(right_map))

    comparisons = []
    for domain in domains:
        left_profile = _find_profile(left_profiles, domain)
        right_profile = _find_profile(right_profiles, domain)
        comparisons.append(
            {
                "domain": domain,
                "left_sample_size": int(left_profile["sample_size"]),
                "right_sample_size": int(right_profile["sample_size"]),
                "left_top_keywords": left_profile["top_keywords"][:5],
                "right_top_keywords": right_profile["top_keywords"][:5],
                "left_duration": left_profile["recommended_duration"],
                "right_duration": right_profile["recommended_duration"],
            }
        )

    return {
        "left_source": left_source,
        "right_source": right_source,
        "comparisons": comparisons,
    }


def build_recommendation_admin_report(db: Session, *, profile_limit: int = 150) -> Dict[str, object]:
    youtube_profiles = build_dataset_profiles(db, source_prefix="youtube", limit=profile_limit)
    google_profiles = build_dataset_profiles(db, source_prefix="google", limit=profile_limit)
    tiktok_profiles = build_dataset_profiles(db, source_prefix="tiktok", limit=profile_limit)

    production_query = production_transcript_query(db)
    total_datasets = production_query.count()
    with_duration = production_query.filter(
        DatasetContent.is_duration_recommendation_eligible.is_(True)
    ).count()
    youtube_count = production_transcript_query(db).filter(
        DatasetContent.source_platform.like("youtube%")
    ).count()
    google_count = production_transcript_query(db).filter(
        DatasetContent.source_platform.like("google%")
    ).count()
    tiktok_count = production_transcript_query(db).filter(
        DatasetContent.source_platform.like("tiktok%")
    ).count()

    recent_sources = (
        production_transcript_query(db)
        .with_entities(DatasetContent.source_platform)
        .order_by(DatasetContent.created_at.desc())
        .limit(50)
        .all()
    )
    recent_source_counts: Dict[str, int] = defaultdict(int)
    for (source_platform,) in recent_sources:
        recent_source_counts[source_platform or "unknown"] += 1

    return {
        "dataset_health": {
            "total_dataset_contents": total_datasets,
            "youtube_dataset_contents": youtube_count,
            "google_dataset_contents": google_count,
            "tiktok_dataset_contents": tiktok_count,
            "duration_coverage_count": with_duration,
            "duration_coverage_ratio": round((with_duration / total_datasets), 3) if total_datasets else 0.0,
        },
        "profile_health": {
            "youtube_profiles": len(youtube_profiles),
            "google_profiles": len(google_profiles),
            "tiktok_profiles": len(tiktok_profiles),
            "youtube_domains": [profile["domain"] for profile in youtube_profiles],
            "google_domains": [profile["domain"] for profile in google_profiles],
            "tiktok_domains": [profile["domain"] for profile in tiktok_profiles],
        },
        "recent_source_activity": [
            {"source_platform": key, "count": value}
            for key, value in sorted(recent_source_counts.items(), key=lambda item: (-item[1], item[0]))
        ],
        "youtube_profiles": youtube_profiles,
        "google_profiles": google_profiles,
        "tiktok_profiles": tiktok_profiles,
    }
