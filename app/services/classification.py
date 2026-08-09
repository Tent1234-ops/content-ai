from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Dict, List

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.database.models import DatasetContent
from app.services.nlp import filter_tokens
from app.services.pipeline.core import normalize_space
from app.services.pipeline.domain_rules import (
    DOMAIN_BASE,
    DOMAIN_HINTS,
    category_query_values,
    detect_domain,
    normalize_domain,
)


def _weighted_dataset_rows(db: Session, source_prefix: str, limit: int) -> List[DatasetContent]:
    domains = list(DOMAIN_BASE)
    per_domain_limit = max(5, math.ceil(limit / max(1, len(domains))))
    rows: List[DatasetContent] = []
    seen_ids: set[int] = set()
    for domain in domains:
        domain_rows = (
            db.query(DatasetContent)
            .filter(DatasetContent.source_platform.like(f"{source_prefix}%"))
            .filter(
                func.lower(func.trim(DatasetContent.category)).in_(
                    category_query_values(domain)
                )
            )
            .order_by(
                DatasetContent.trend_score.desc(),
                DatasetContent.views.desc(),
                DatasetContent.created_at.desc(),
            )
            .limit(per_domain_limit)
            .all()
        )
        for row in domain_rows:
            if row.dataset_id in seen_ids:
                continue
            seen_ids.add(row.dataset_id)
            rows.append(row)
    return rows


def _token_counter(text: str) -> Counter[str]:
    normalized = normalize_space(text or "").lower()
    tokens = re.findall(r"[\u0E00-\u0E7Fa-z0-9][\u0E00-\u0E7Fa-z0-9\-]*", normalized)
    return Counter(filter_tokens(tokens))


def _cosine(left: Counter[str] | Dict[str, float], right: Counter[str] | Dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    shared = set(left) & set(right)
    numerator = sum(float(left[key]) * float(right[key]) for key in shared)
    left_norm = math.sqrt(sum(float(value) ** 2 for value in left.values()))
    right_norm = math.sqrt(sum(float(value) ** 2 for value in right.values()))
    if left_norm <= 0 or right_norm <= 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def build_domain_classifier_profiles(
    db: Session,
    *,
    source_prefix: str = "youtube",
    limit: int = 200,
) -> List[Dict[str, object]]:
    rows = _weighted_dataset_rows(db, source_prefix, limit)
    term_profiles: Dict[str, Counter[str]] = defaultdict(Counter)
    sample_counts: Counter[str] = Counter()

    for row in rows:
        text = " ".join(part for part in [row.title or "", row.transcript or "", row.category or ""] if part).strip()
        if not text:
            continue
        domain = normalize_domain(row.category)
        if domain == "general":
            domain = detect_domain(text)
        weight = 1.0 + min(float(row.trend_score or 0.0) / 500000.0, 5.0)
        for term, count in _token_counter(text).items():
            term_profiles[domain][term] += count * weight
        sample_counts[domain] += 1

    for domain in DOMAIN_BASE:
        term_profiles.setdefault(domain, Counter())

    profiles: List[Dict[str, object]] = []
    for domain, terms in term_profiles.items():
        seed_terms = DOMAIN_BASE.get(domain, []) + DOMAIN_HINTS.get(domain, [])
        for seed in seed_terms:
            terms[seed] += 0.15
            for token in _token_counter(seed):
                terms[token] += 0.75
        profiles.append(
            {
                "domain": domain,
                "sample_size": int(sample_counts[domain]),
                "top_terms": [
                    {"term": term, "weight": round(float(weight), 3)}
                    for term, weight in terms.most_common(12)
                ],
                "term_weights": dict(terms),
            }
        )

    profiles.sort(key=lambda item: (-int(item["sample_size"]), str(item["domain"])))
    return profiles


def classify_text_domain(
    db: Session,
    *,
    text: str,
    title: str | None = None,
    source_prefix: str = "youtube",
    profile_limit: int = 200,
    top_k: int = 5,
) -> Dict[str, object]:
    merged_text = " ".join(part for part in [title or "", text or ""] if part).strip()
    rule_domain = detect_domain(merged_text)
    input_terms = _token_counter(merged_text)
    profiles = build_domain_classifier_profiles(db, source_prefix=source_prefix, limit=profile_limit)

    candidates: List[Dict[str, object]] = []
    for profile in profiles:
        domain = str(profile["domain"])
        term_weights = profile["term_weights"]
        similarity = _cosine(input_terms, term_weights)
        rule_bonus = 0.08 if domain == rule_domain and rule_domain != "general" else 0.0
        score = similarity + rule_bonus
        matched_terms = [
            term
            for term, _weight in Counter(term_weights).most_common(20)
            if term in input_terms
        ][:8]
        candidates.append(
            {
                "domain": domain,
                "score": round(score, 4),
                "similarity": round(similarity, 4),
                "sample_size": int(profile["sample_size"]),
                "matched_terms": matched_terms,
            }
        )

    if not candidates:
        fallback_terms = DOMAIN_BASE.get(rule_domain, [])[:8]
        return {
            "domain": rule_domain,
            "confidence": 0.55 if rule_domain != "general" else 0.25,
            "method": "rule_fallback",
            "rule_domain": rule_domain,
            "source": source_prefix,
            "profile_limit": profile_limit,
            "candidates": [
                {
                    "domain": rule_domain,
                    "score": 0.0,
                    "similarity": 0.0,
                    "sample_size": 0,
                    "matched_terms": fallback_terms,
                }
            ],
        }

    candidates.sort(key=lambda item: (-float(item["score"]), -int(item["sample_size"]), str(item["domain"])))
    top_score = float(candidates[0]["score"])
    score_total = sum(max(float(item["score"]), 0.0) for item in candidates[:top_k])
    confidence = top_score / score_total if score_total > 0 else 0.0
    if candidates[0]["domain"] == rule_domain and rule_domain != "general":
        confidence = min(1.0, confidence + 0.08)

    return {
        "domain": candidates[0]["domain"],
        "confidence": round(confidence, 4),
        "method": "dataset_centroid_cosine",
        "rule_domain": rule_domain,
        "source": source_prefix,
        "profile_limit": profile_limit,
        "candidates": candidates[:top_k],
    }
