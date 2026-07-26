from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
from statistics import median
from typing import Dict, Iterable, List

from sqlalchemy.orm import Session

from app.database.models import DatasetContent, UserContent
from app.services.admin_settings import get_admin_config
from app.services.classification import classify_text_domain
from app.services.nlp import filter_tokens, normalize_text_for_nlp, run_nlp_pipeline, tokenize_text
from app.services.pipeline.core import (
    build_comparison_profile,
    build_dimension_status,
    extract_features,
    feature_to_keywords,
    normalize_asr_terms,
)
from app.services.pipeline.domain_rules import (
    DOMAIN_BASE,
    DOMAIN_DIMENSIONS_ORDER,
    detect_domain,
    infer_domain_from_features,
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
}


def _weighted_increment(counter: Dict[str, float], key: str, weight: float) -> None:
    if not key:
        return
    counter[key] = counter.get(key, 0.0) + weight


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

    min_date = datetime.utcnow() - timedelta(days=admin_config.analysis_time_range_days)
    rows = (
        db.query(DatasetContent)
        .filter(DatasetContent.source_platform.like(f"{source_prefix}%"))
        .filter(DatasetContent.created_at >= min_date)
        .order_by(DatasetContent.trend_score.desc(), DatasetContent.views.desc(), DatasetContent.created_at.desc())
        .limit(limit)
        .all()
    )

    grouped_rows: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    keyword_scores: Dict[str, Dict[str, float]] = defaultdict(dict)
    dimension_scores: Dict[str, Dict[str, float]] = defaultdict(dict)
    hook_scores: Dict[str, Dict[str, float]] = defaultdict(dict)
    domain_durations: Dict[str, List[int]] = defaultdict(list)

    for row in rows:
        base_text = " ".join(part for part in [row.title or "", row.transcript or "", row.category or ""] if part).strip()
        if not base_text:
            continue
        snapshot = analyze_text_snapshot(
            base_text,
            row.title,
            max_keywords=admin_config.max_keywords_display,
            hook_duration=admin_config.hook_analysis_duration,
        )
        domain = str(snapshot["domain"])
        weight = 1.0 + min(float(row.trend_score or 0.0) / 500000.0, 5.0)
        grouped_rows[domain].append({"row": row, "snapshot": snapshot, "weight": weight})

        for item in snapshot["nlp_result"].get("top_keywords", []):
            _weighted_increment(keyword_scores[domain], item["keyword"], weight * float(item["score"]))
        for item in snapshot["comparison_dimensions"]:
            _weighted_increment(
                dimension_scores[domain],
                item["name"],
                weight * float(item.get("confidence", 0.0)),
            )
        for term in snapshot["hook_terms"]:
            _weighted_increment(hook_scores[domain], term, weight)
        if row.duration_seconds:
            domain_durations[domain].append(int(row.duration_seconds))

    profiles: List[Dict[str, object]] = []
    for domain, items in grouped_rows.items():
        if not items:
            continue
        top_keywords = [
            {"keyword": keyword, "score": round(score, 3)}
            for keyword, score in sorted(keyword_scores[domain].items(), key=lambda item: (-item[1], item[0]))[:10]
        ]
        top_dimensions = [
            {"keyword": name, "score": round(score, 3)}
            for name, score in sorted(dimension_scores[domain].items(), key=lambda item: (-item[1], item[0]))[:8]
        ]
        hook_keywords = [
            {"keyword": keyword, "score": round(score, 3)}
            for keyword, score in sorted(hook_scores[domain].items(), key=lambda item: (-item[1], item[0]))[:8]
        ]
        exemplar_titles = [item["row"].title for item in items[:3]]
        profiles.append(
            {
                "domain": domain,
                "sample_size": len(items),
                "top_keywords": top_keywords,
                "top_dimensions": top_dimensions,
                "hook_keywords": hook_keywords,
                "recommended_duration": _duration_summary(domain_durations[domain], domain),
                "exemplar_titles": exemplar_titles,
            }
        )

    profiles.sort(key=lambda item: (item["domain"] == "general", -item["sample_size"], item["domain"]))
    return profiles


def _find_profile(profiles: Iterable[Dict[str, object]], domain: str) -> Dict[str, object]:
    for profile in profiles:
        if profile["domain"] == domain:
            return profile
    return {
        "domain": domain,
        "sample_size": 0,
        "top_keywords": [
            {"keyword": keyword, "score": 0.0}
            for keyword in DOMAIN_BASE.get(domain, [])[:8]
        ],
        "top_dimensions": [
            {"keyword": name, "score": 0.0}
            for name in DOMAIN_DIMENSIONS_ORDER.get(domain, [])[:8]
        ],
        "hook_keywords": [],
        "recommended_duration": _duration_summary([], domain),
        "exemplar_titles": [],
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
    if float(classification.get("confidence", 0.0)) >= 0.45:
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
    profiles = build_dataset_profiles(db, source_prefix=source_prefix, limit=profile_limit)
    profile = _find_profile(profiles, domain)

    normalized_user_keywords = {keyword.lower() for keyword in user_keywords}
    missing_keywords = []
    for item in profile["top_keywords"]:
        keyword = item["keyword"]
        if keyword.lower() in normalized_user_keywords:
            continue
        missing_keywords.append({"keyword": keyword, "score": round(float(item["score"]), 3)})
        if len(missing_keywords) >= 6:
            break

    normalized_hook_terms = {term.lower() for term in hook_terms}
    hook_keywords = []
    for item in profile["hook_keywords"]:
        keyword = item["keyword"]
        if keyword.lower() in normalized_hook_terms or keyword.lower() in normalized_user_keywords:
            continue
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

    total_datasets = db.query(DatasetContent).count()
    with_duration = db.query(DatasetContent).filter(DatasetContent.duration_seconds.isnot(None)).count()
    youtube_count = db.query(DatasetContent).filter(DatasetContent.source_platform.like("youtube%")).count()
    google_count = db.query(DatasetContent).filter(DatasetContent.source_platform.like("google%")).count()

    recent_sources = (
        db.query(DatasetContent.source_platform)
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
            "duration_coverage_count": with_duration,
            "duration_coverage_ratio": round((with_duration / total_datasets), 3) if total_datasets else 0.0,
        },
        "profile_health": {
            "youtube_profiles": len(youtube_profiles),
            "google_profiles": len(google_profiles),
            "youtube_domains": [profile["domain"] for profile in youtube_profiles],
            "google_domains": [profile["domain"] for profile in google_profiles],
        },
        "recent_source_activity": [
            {"source_platform": key, "count": value}
            for key, value in sorted(recent_source_counts.items(), key=lambda item: (-item[1], item[0]))
        ],
        "youtube_profiles": youtube_profiles,
        "google_profiles": google_profiles,
    }
