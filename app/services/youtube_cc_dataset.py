from __future__ import annotations

import csv
import hashlib
import html
import io
import json
import math
import re
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from sqlalchemy.orm import Session

from app.database.models import (
    DatasetCollectionRun,
    DatasetContent,
    DatasetReviewEvent,
)
from app.services.dataset_contract import (
    ACCEPTED_TRANSCRIPT_QUALITIES,
    DEFAULT_YOUTUBE_CC_DATASET_VERSION,
    RECOMMENDATION_DURATION_MAX_SECONDS,
    SPLIT_STRATEGY,
    SUPPORTED_CAPTION_TYPES,
    SUPPORTED_TRANSCRIPT_LANGUAGES,
    TRANSCRIPT_WINDOW_SECONDS,
    YOUTUBE_CC_DATASET_SOURCE,
    YOUTUBE_CC_LABEL_SOURCE,
    YOUTUBE_CC_LICENSE_NAME,
    YOUTUBE_CC_LICENSE_URL,
    YOUTUBE_CC_TRANSCRIPT_SOURCE,
    YOUTUBE_CC_VERIFICATION_STATUS,
    channel_dataset_split,
)
from app.services.dataset_eligibility import validate_training_eligibility_values
from app.services.taxonomy import (
    ACTIVE_LEAF_KEYS,
    TAXONOMY_VERSION,
    UNKNOWN_LEAF_KEY,
    collection_queries_for_leaf,
    normalize_taxonomy_leaf,
    sync_taxonomy_registry,
    taxonomy_coverage,
    taxonomy_path,
)


YOUTUBE_API_BASE = "https://www.googleapis.com/youtube/v3"
CLASSIFICATION_DIVERSE_STRATEGY = "classification_diverse"
RECOMMENDATION_HIGH_PERFORMANCE_STRATEGY = "recommendation_high_performance"
COLLECTION_STRATEGIES = (
    CLASSIFICATION_DIVERSE_STRATEGY,
    RECOMMENDATION_HIGH_PERFORMANCE_STRATEGY,
)
COLLECTION_SCHEMA_VERSION = 4
DEFAULT_THAI_TARGET_RATIO = 0.40
DEFAULT_MAX_VIDEOS_PER_CHANNEL_PER_LEAF = 3
REVIEW_FIELDS = (
    "candidate_sha256",
    "source_youtube_id",
    "title",
    "video_url",
    "channel_title",
    "proposed_leaf_key",
    "transcript_language",
    "caption_type",
    "duration_seconds",
    "transcript_preview",
    "decision",
    "reviewed_leaf_key",
    "transcript_quality",
    "reviewer",
    "reviewed_at",
    "review_notes",
)


class YouTubeCCDatasetError(RuntimeError):
    pass


class YouTubeQuotaExceededError(YouTubeCCDatasetError):
    def __init__(self, resource: str, status_code: int, detail: str):
        self.resource = resource
        self.status_code = status_code
        self.detail = detail
        super().__init__(
            f"YouTube API {resource} quota exhausted (HTTP {status_code}): "
            f"{detail[:500]}"
        )


class YouTubeTranscriptProviderBlockedError(YouTubeCCDatasetError):
    def __init__(self, video_id: str, detail: str):
        self.video_id = video_id
        self.detail = detail
        super().__init__(
            "YouTube transcript provider blocked this IP while fetching "
            f"{video_id}: {detail[:500]}"
        )


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _iso_z(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_datetime(value: Any, *, field: str) -> datetime:
    raw = str(value or "").strip()
    if not raw:
        raise YouTubeCCDatasetError(f"Missing {field}")
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise YouTubeCCDatasetError(f"Invalid {field}: {raw}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).replace(tzinfo=None)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(text, encoding="utf-8", newline="")
    temp_path.replace(path)


def parse_iso8601_duration(value: str) -> int:
    match = re.fullmatch(
        r"P(?:(?P<days>\d+)D)?T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?",
        str(value or ""),
    )
    if not match:
        raise YouTubeCCDatasetError(f"Unsupported YouTube duration: {value!r}")
    parts = {key: int(item or 0) for key, item in match.groupdict().items()}
    return (
        parts["days"] * 86400
        + parts["hours"] * 3600
        + parts["minutes"] * 60
        + parts["seconds"]
    )


def _youtube_get(
    resource: str,
    *,
    api_key: str,
    timeout_seconds: float,
    **params: Any,
) -> dict[str, Any]:
    if not api_key:
        raise YouTubeCCDatasetError("YOUTUBE_API_KEY is required")
    query = urllib.parse.urlencode({**params, "key": api_key})
    request = urllib.request.Request(
        f"{YOUTUBE_API_BASE}/{resource}?{query}",
        headers={"User-Agent": "content-ai-dataset-collector/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        quota_markers = (
            "quota exceeded",
            "quotaexceeded",
            "dailylimitexceeded",
            "ratelimitexceeded",
        )
        if exc.code in (403, 429) and any(
            marker in detail.lower() for marker in quota_markers
        ):
            raise YouTubeQuotaExceededError(resource, exc.code, detail) from exc
        raise YouTubeCCDatasetError(
            f"YouTube API {resource} failed with HTTP {exc.code}: {detail[:500]}"
        ) from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise YouTubeCCDatasetError(f"YouTube API {resource} failed: {exc}") from exc


def _transcript_segments(raw_segments: Iterable[Any]) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for raw in raw_segments:
        if isinstance(raw, dict):
            text = raw.get("text")
            start = raw.get("start")
            duration = raw.get("duration")
        else:
            text = getattr(raw, "text", None)
            start = getattr(raw, "start", None)
            duration = getattr(raw, "duration", None)
        try:
            start_value = float(start or 0.0)
            duration_value = max(0.0, float(duration or 0.0))
        except (TypeError, ValueError):
            continue
        if start_value >= TRANSCRIPT_WINDOW_SECONDS:
            continue
        clean_text = html.unescape(str(text or ""))
        clean_text = re.sub(r"\s+", " ", clean_text).strip()
        if not clean_text:
            continue
        segments.append(
            {
                "text": clean_text,
                "start": round(start_value, 3),
                "duration": round(
                    min(duration_value, TRANSCRIPT_WINDOW_SECONDS - start_value),
                    3,
                ),
            }
        )
    return segments


def fetch_public_transcript(
    video_id: str,
    languages: Sequence[str],
) -> dict[str, Any]:
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        from youtube_transcript_api._errors import IpBlocked, RequestBlocked
    except ImportError as exc:
        raise YouTubeCCDatasetError(
            "youtube-transcript-api is not installed; run pip install -r requirements.txt"
        ) from exc

    try:
        tracks = list(YouTubeTranscriptApi().list(video_id))
    except (IpBlocked, RequestBlocked) as exc:
        raise YouTubeTranscriptProviderBlockedError(video_id, str(exc)) from exc
    except Exception as exc:
        raise YouTubeCCDatasetError(f"Transcript list failed for {video_id}: {exc}") from exc

    language_order = {language: index for index, language in enumerate(languages)}
    candidates = []
    for track in tracks:
        language_code = str(getattr(track, "language_code", "")).lower()
        base_language = language_code.split("-", 1)[0]
        if base_language not in language_order:
            continue
        is_generated = bool(getattr(track, "is_generated", False))
        candidates.append(
            (
                language_order[base_language],
                1 if is_generated else 0,
                track,
                base_language,
                is_generated,
            )
        )
    if not candidates:
        raise YouTubeCCDatasetError(
            f"No transcript in supported languages {list(languages)} for {video_id}"
        )

    _language_rank, _generated_rank, selected, language, is_generated = min(
        candidates,
        key=lambda item: (item[0], item[1]),
    )
    try:
        fetched = selected.fetch()
        raw_segments = (
            fetched.to_raw_data()
            if hasattr(fetched, "to_raw_data")
            else list(fetched)
        )
    except (IpBlocked, RequestBlocked) as exc:
        raise YouTubeTranscriptProviderBlockedError(video_id, str(exc)) from exc
    except Exception as exc:
        raise YouTubeCCDatasetError(f"Transcript fetch failed for {video_id}: {exc}") from exc

    segments = _transcript_segments(raw_segments)
    transcript = " ".join(segment["text"] for segment in segments).strip()
    if not transcript:
        raise YouTubeCCDatasetError(
            f"Transcript has no text in the first {TRANSCRIPT_WINDOW_SECONDS} seconds"
        )
    end_seconds = max(
        segment["start"] + segment["duration"] for segment in segments
    )
    return {
        "language": language,
        "caption_type": "auto_generated" if is_generated else "manual",
        "segments": segments,
        "segment_count": len(segments),
        "start_seconds": float(segments[0]["start"]),
        "end_seconds": round(min(end_seconds, TRANSCRIPT_WINDOW_SECONDS), 3),
        "transcript": transcript,
        "transcript_sha256": _sha256_text(transcript),
    }


def _candidate_hash(candidate: dict[str, Any]) -> str:
    payload = {key: value for key, value in candidate.items() if key != "candidate_sha256"}
    return _sha256_text(_canonical_json(payload))


def _performance_metrics(candidate: dict[str, Any]) -> dict[str, float]:
    views = max(0, int(candidate.get("views") or 0))
    likes = max(0, int(candidate.get("likes") or 0))
    comments = max(0, int(candidate.get("comments") or 0))
    engagement_rate = (likes + comments) / max(views, 1)
    try:
        published = _parse_datetime(candidate.get("published_at"), field="published_at")
        captured = _parse_datetime(
            candidate.get("statistics_captured_at"),
            field="statistics_captured_at",
        )
        age_days = max((captured - published).total_seconds() / 86400.0, 1.0)
    except YouTubeCCDatasetError:
        age_days = 1.0
    average_views_per_day = views / age_days
    performance_signal = math.log1p(average_views_per_day) * (
        1.0 + min(engagement_rate * 10.0, 1.0)
    )
    return {
        "average_views_per_day": round(average_views_per_day, 6),
        "engagement_rate": round(engagement_rate, 8),
        "performance_signal": round(performance_signal, 6),
    }


def _make_candidate(
    *,
    item: dict[str, Any],
    transcript: dict[str, Any],
    run_key: str,
    dataset_version: str,
    leaf_key: str,
    query: str,
    region_code: str,
    collected_at: datetime,
    collection_strategy: str,
    search_order: str,
    search_rank: int,
) -> dict[str, Any]:
    snippet = item.get("snippet") or {}
    content_details = item.get("contentDetails") or {}
    statistics = item.get("statistics") or {}
    status = item.get("status") or {}
    video_id = str(item.get("id") or "").strip()
    duration_seconds = parse_iso8601_duration(str(content_details.get("duration") or ""))
    candidate = {
        "schema_version": COLLECTION_SCHEMA_VERSION,
        "run_key": run_key,
        "dataset_source": YOUTUBE_CC_DATASET_SOURCE,
        "dataset_version": dataset_version,
        "taxonomy_version": TAXONOMY_VERSION,
        "proposed_leaf_key": leaf_key,
        "collection_query": query,
        "collection_strategy": collection_strategy,
        "search_order": search_order,
        "search_rank": search_rank,
        "region_code": region_code,
        "collected_at": _iso_z(collected_at),
        "statistics_captured_at": _iso_z(collected_at),
        "license_verified_at": _iso_z(collected_at),
        "source_youtube_id": video_id,
        "source_record_id": video_id,
        "title": html.unescape(str(snippet.get("title") or "")).strip(),
        "description": html.unescape(str(snippet.get("description") or "")).strip(),
        "video_url": f"https://www.youtube.com/watch?v={video_id}",
        "channel_id": str(snippet.get("channelId") or "").strip(),
        "channel_title": html.unescape(str(snippet.get("channelTitle") or "")).strip(),
        "youtube_category_id": str(snippet.get("categoryId") or "").strip(),
        "published_at": str(snippet.get("publishedAt") or ""),
        "duration_seconds": duration_seconds,
        "views": int(statistics.get("viewCount") or 0),
        "likes": int(statistics.get("likeCount") or 0),
        "comments": int(statistics.get("commentCount") or 0),
        "youtube_license_code": str(status.get("license") or ""),
        "license_name": YOUTUBE_CC_LICENSE_NAME,
        "license_url": YOUTUBE_CC_LICENSE_URL,
        "transcript_source": YOUTUBE_CC_TRANSCRIPT_SOURCE,
        "transcript_language": transcript["language"],
        "caption_type": transcript["caption_type"],
        "transcript": transcript["transcript"],
        "transcript_sha256": transcript["transcript_sha256"],
        "transcript_segments": transcript["segments"],
        "transcript_segment_count": transcript["segment_count"],
        "transcript_start_seconds": transcript["start_seconds"],
        "transcript_end_seconds": transcript["end_seconds"],
        "transcript_window_seconds": TRANSCRIPT_WINDOW_SECONDS,
        "raw_metadata": {
            "snippet": snippet,
            "contentDetails": content_details,
            "statistics": statistics,
            "status": status,
        },
    }
    candidate.update(_performance_metrics(candidate))
    candidate["candidate_sha256"] = _candidate_hash(candidate)
    return candidate


def _review_csv_text(candidates: Sequence[dict[str, Any]]) -> str:
    import io

    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=REVIEW_FIELDS, lineterminator="\n")
    writer.writeheader()
    for candidate in candidates:
        transcript = re.sub(r"\s+", " ", str(candidate["transcript"])).strip()
        writer.writerow(
            {
                "candidate_sha256": candidate["candidate_sha256"],
                "source_youtube_id": candidate["source_youtube_id"],
                "title": candidate["title"],
                "video_url": candidate["video_url"],
                "channel_title": candidate["channel_title"],
                "proposed_leaf_key": candidate["proposed_leaf_key"],
                "transcript_language": candidate["transcript_language"],
                "caption_type": candidate["caption_type"],
                "duration_seconds": candidate["duration_seconds"],
                "transcript_preview": transcript[:500],
                "decision": "",
                "reviewed_leaf_key": "",
                "transcript_quality": "",
                "reviewer": "",
                "reviewed_at": "",
                "review_notes": "",
            }
        )
    return buffer.getvalue()


def _sampling_plan(
    target_per_leaf: int,
    performance_target_per_leaf: int | None,
) -> list[dict[str, Any]]:
    if target_per_leaf < 1:
        raise YouTubeCCDatasetError("target_per_leaf must be at least 1")
    if performance_target_per_leaf is None:
        performance_target_per_leaf = min(
            target_per_leaf,
            max(1, math.ceil(target_per_leaf * 0.30)),
        )
    if not 0 <= performance_target_per_leaf <= target_per_leaf:
        raise YouTubeCCDatasetError(
            "performance_target_per_leaf must be between 0 and target_per_leaf"
        )

    classification_target = target_per_leaf - performance_target_per_leaf
    plan: list[dict[str, Any]] = []
    if performance_target_per_leaf:
        plan.append(
            {
                "strategy": RECOMMENDATION_HIGH_PERFORMANCE_STRATEGY,
                "search_order": "viewCount",
                "target_per_leaf": performance_target_per_leaf,
            }
        )
    if classification_target:
        plan.append(
            {
                "strategy": CLASSIFICATION_DIVERSE_STRATEGY,
                "search_order": "relevance",
                "target_per_leaf": classification_target,
            }
        )
    return plan


def _quality_collection_settings(
    *,
    target_per_leaf: int,
    languages: Sequence[str],
    min_thai_per_leaf: int | None,
    max_videos_per_channel_per_leaf: int,
) -> tuple[int, int]:
    if min_thai_per_leaf is None:
        min_thai_per_leaf = (
            math.ceil(target_per_leaf * DEFAULT_THAI_TARGET_RATIO)
            if "th" in languages
            else 0
        )
    if not 0 <= min_thai_per_leaf <= target_per_leaf:
        raise YouTubeCCDatasetError(
            "min_thai_per_leaf must be between 0 and target_per_leaf"
        )
    if min_thai_per_leaf and "th" not in languages:
        raise YouTubeCCDatasetError(
            "min_thai_per_leaf requires 'th' in the selected languages"
        )
    if max_videos_per_channel_per_leaf < 1:
        raise YouTubeCCDatasetError(
            "max_videos_per_channel_per_leaf must be at least 1"
        )
    return min_thai_per_leaf, max_videos_per_channel_per_leaf


def _queries_by_leaf(leaf_keys: Sequence[str]) -> dict[str, list[str]]:
    return {
        leaf_key: list(collection_queries_for_leaf(leaf_key))
        for leaf_key in leaf_keys
    }


def _upgrade_empty_run_quality_config(
    collection_run: DatasetCollectionRun,
    run_config: dict[str, Any],
) -> dict[str, Any]:
    if (
        "min_thai_per_leaf" in run_config
        and "max_videos_per_channel_per_leaf" in run_config
        and "queries_by_leaf" in run_config
    ):
        return run_config
    if int(collection_run.transcripts_collected or 0) > 0:
        raise YouTubeCCDatasetError(
            "This run started before language/channel quality controls and already "
            "contains candidates; start a new run to keep collection reproducible"
        )

    upgraded = dict(run_config)
    target_per_leaf = int(upgraded["target_per_leaf"])
    languages = tuple(upgraded["languages"])
    min_thai, channel_cap = _quality_collection_settings(
        target_per_leaf=target_per_leaf,
        languages=languages,
        min_thai_per_leaf=None,
        max_videos_per_channel_per_leaf=(
            DEFAULT_MAX_VIDEOS_PER_CHANNEL_PER_LEAF
        ),
    )
    upgraded.update(
        {
            "schema_version": COLLECTION_SCHEMA_VERSION,
            "min_thai_per_leaf": min_thai,
            "max_videos_per_channel_per_leaf": channel_cap,
            "language_balance_policy": "reserve_minimum_thai_v1",
            "channel_diversity_policy": "max_per_channel_per_leaf_v1",
            "queries_by_leaf": _queries_by_leaf(upgraded["leaf_keys"]),
        }
    )
    collection_run.run_key = _sha256_text(_canonical_json(upgraded))
    return upgraded


def _normalize_collection_inputs(
    *,
    leaf_keys: Sequence[str],
    languages: Sequence[str],
    max_pages_per_query: int,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    normalized_leaves = tuple(
        dict.fromkeys(normalize_taxonomy_leaf(item) for item in leaf_keys)
    )
    if not normalized_leaves or any(
        item not in ACTIVE_LEAF_KEYS for item in normalized_leaves
    ):
        raise YouTubeCCDatasetError("All requested leaves must be active taxonomy leaves")
    normalized_languages = tuple(
        dict.fromkeys(str(item).lower() for item in languages)
    )
    if not normalized_languages or any(
        item not in SUPPORTED_TRANSCRIPT_LANGUAGES
        for item in normalized_languages
    ):
        raise YouTubeCCDatasetError(
            f"Languages must be selected from {SUPPORTED_TRANSCRIPT_LANGUAGES}"
        )
    if max_pages_per_query < 1:
        raise YouTubeCCDatasetError("max_pages_per_query must be at least 1")
    return normalized_leaves, normalized_languages


def _resolve_project_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    return path


def _load_collection_manifest(
    collection_run: DatasetCollectionRun,
    *,
    required: bool,
) -> dict[str, Any]:
    raw_path = str(collection_run.manifest_path or "").strip()
    if not raw_path:
        if required:
            raise YouTubeCCDatasetError(
                f"Collection run {collection_run.collection_run_id} has no manifest"
            )
        return {}
    path = _resolve_project_path(raw_path)
    if not path.is_file():
        if required:
            raise YouTubeCCDatasetError(f"Collection manifest not found: {path}")
        return {}
    actual_hash = _sha256_file(path)
    if collection_run.manifest_sha256 and collection_run.manifest_sha256 != actual_hash:
        raise YouTubeCCDatasetError(
            f"Manifest hash mismatch for run {collection_run.collection_run_id}"
        )
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise YouTubeCCDatasetError(f"Invalid collection manifest: {path}") from exc


def repair_quota_waiting_run_statuses(db: Session) -> dict[str, Any]:
    repaired_ids: list[int] = []
    warnings: list[str] = []
    runs = (
        db.query(DatasetCollectionRun)
        .filter(
            DatasetCollectionRun.dataset_source == YOUTUBE_CC_DATASET_SOURCE,
            DatasetCollectionRun.status == "failed",
        )
        .all()
    )
    for run in runs:
        try:
            manifest = _load_collection_manifest(run, required=False)
        except YouTubeCCDatasetError as exc:
            warnings.append(f"run {run.collection_run_id}: {exc}")
            continue
        failure = str(manifest.get("failure") or "").lower()
        if not (
            "http 429" in failure
            or "quota exceeded" in failure
            or "quota exhausted" in failure
        ):
            continue
        run.status = "quota_waiting"
        run.completed_at = None
        if run.errors_count and run.errors_count > 0:
            run.errors_count -= 1
        repaired_ids.append(run.collection_run_id)
    if repaired_ids:
        db.commit()
    return {"repaired": len(repaired_ids), "run_ids": repaired_ids, "warnings": warnings}


def _dedup_catalog(
    db: Session,
    *,
    exclude_run_id: int,
) -> tuple[set[str], set[str], Counter[tuple[str, str]], dict[str, Any]]:
    video_ids: set[str] = set()
    transcript_hashes: set[str] = set()
    channel_leaf_counts: Counter[tuple[str, str]] = Counter()
    database_rows = (
        db.query(
            DatasetContent.source_youtube_id,
            DatasetContent.transcript_sha256,
            DatasetContent.taxonomy_leaf_key,
            DatasetContent.source_channel_id,
        )
        .filter(DatasetContent.source_youtube_id.isnot(None))
        .all()
    )
    for video_id, transcript_hash, leaf_key, channel_id in database_rows:
        normalized_video_id = str(video_id or "").strip()
        normalized_transcript_hash = str(transcript_hash or "").strip()
        normalized_leaf = normalize_taxonomy_leaf(leaf_key)
        normalized_channel = str(channel_id or "").strip()
        if normalized_video_id:
            video_ids.add(normalized_video_id)
        if normalized_transcript_hash:
            transcript_hashes.add(normalized_transcript_hash)
        if normalized_leaf in ACTIVE_LEAF_KEYS and normalized_channel:
            channel_leaf_counts[(normalized_leaf, normalized_channel)] += 1
    database_video_count = len(video_ids)
    database_transcript_count = len(transcript_hashes)
    artifact_rows = 0
    artifact_runs = 0
    warnings: list[str] = []

    runs = (
        db.query(DatasetCollectionRun)
        .filter(
            DatasetCollectionRun.dataset_source == YOUTUBE_CC_DATASET_SOURCE,
            DatasetCollectionRun.collection_run_id != exclude_run_id,
        )
        .all()
    )
    for run in runs:
        if not str(run.candidate_artifact_path or "").strip():
            continue
        try:
            candidates = _load_candidates(
                _candidate_artifact_for_run(run),
                allow_empty=True,
            )
        except YouTubeCCDatasetError as exc:
            warnings.append(f"run {run.collection_run_id}: {exc}")
            continue
        artifact_runs += 1
        artifact_rows += len(candidates)
        for candidate in candidates.values():
            video_id = str(candidate.get("source_youtube_id") or "").strip()
            transcript_hash = str(candidate.get("transcript_sha256") or "").strip()
            is_new_video = bool(video_id and video_id not in video_ids)
            if is_new_video:
                video_ids.add(video_id)
                leaf_key = normalize_taxonomy_leaf(
                    candidate.get("proposed_leaf_key")
                )
                channel_id = str(candidate.get("channel_id") or "").strip()
                if leaf_key in ACTIVE_LEAF_KEYS and channel_id:
                    channel_leaf_counts[(leaf_key, channel_id)] += 1
            if transcript_hash:
                transcript_hashes.add(transcript_hash)

    channels_by_leaf: dict[str, set[str]] = defaultdict(set)
    for leaf_key, channel_id in channel_leaf_counts:
        channels_by_leaf[leaf_key].add(channel_id)

    return video_ids, transcript_hashes, channel_leaf_counts, {
        "database_video_ids": database_video_count,
        "database_transcript_hashes": database_transcript_count,
        "prior_artifact_runs": artifact_runs,
        "prior_artifact_rows": artifact_rows,
        "catalog_video_ids": len(video_ids),
        "catalog_transcript_hashes": len(transcript_hashes),
        "catalog_unique_channels_by_leaf": {
            leaf_key: len(channel_ids)
            for leaf_key, channel_ids in channels_by_leaf.items()
        },
        "warnings": warnings,
    }


def _annotate_performance_ranks(candidates: Sequence[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        candidate.update(_performance_metrics(candidate))
        grouped[normalize_taxonomy_leaf(candidate.get("proposed_leaf_key"))].append(
            candidate
        )
    for rows in grouped.values():
        ranked = sorted(
            rows,
            key=lambda item: (
                -float(item.get("performance_signal") or 0.0),
                -int(item.get("views") or 0),
                str(item.get("source_youtube_id") or ""),
            ),
        )
        total = len(ranked)
        for rank, candidate in enumerate(ranked, start=1):
            candidate["performance_rank_within_leaf"] = rank
            candidate["performance_percentile_within_leaf"] = round(
                100.0 * (total - rank + 1) / max(total, 1),
                2,
            )
    for candidate in candidates:
        candidate["candidate_sha256"] = _candidate_hash(candidate)


def _accepted_counts(
    candidates: Sequence[dict[str, Any]],
) -> dict[str, Counter[str]]:
    counts = {strategy: Counter() for strategy in COLLECTION_STRATEGIES}
    for candidate in candidates:
        strategy = str(candidate.get("collection_strategy") or "").strip()
        if strategy not in counts:
            strategy = CLASSIFICATION_DIVERSE_STRATEGY
            candidate["collection_strategy"] = strategy
            candidate["search_order"] = str(candidate.get("search_order") or "relevance")
        leaf_key = normalize_taxonomy_leaf(candidate.get("proposed_leaf_key"))
        counts[strategy][leaf_key] += 1
    return counts


def _language_counts_by_leaf(
    candidates: Sequence[dict[str, Any]],
) -> dict[str, Counter[str]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for candidate in candidates:
        leaf_key = normalize_taxonomy_leaf(candidate.get("proposed_leaf_key"))
        language = str(candidate.get("transcript_language") or "und").lower()
        language = language.split("-", 1)[0]
        counts[leaf_key][language] += 1
    return counts


def _channel_counts_by_leaf(
    candidates: Sequence[dict[str, Any]],
) -> tuple[Counter[tuple[str, str]], dict[str, set[str]]]:
    counts: Counter[tuple[str, str]] = Counter()
    unique_channels: dict[str, set[str]] = defaultdict(set)
    for candidate in candidates:
        leaf_key = normalize_taxonomy_leaf(candidate.get("proposed_leaf_key"))
        channel_id = str(candidate.get("channel_id") or "").strip()
        if leaf_key in ACTIVE_LEAF_KEYS and channel_id:
            counts[(leaf_key, channel_id)] += 1
            unique_channels[leaf_key].add(channel_id)
    return counts, unique_channels


def _quality_values_for_progress(
    run_config: dict[str, Any],
    *,
    candidate_count: int,
) -> tuple[int, int]:
    target = max(1, int(run_config.get("target_per_leaf") or 1))
    languages = tuple(run_config.get("languages") or ())
    if "min_thai_per_leaf" in run_config:
        min_thai = int(run_config["min_thai_per_leaf"])
    elif (
        not candidate_count
        and run_config.get("source_video_duration_policy")
        == "unrestricted_positive_duration"
    ):
        min_thai = (
            math.ceil(target * DEFAULT_THAI_TARGET_RATIO)
            if "th" in languages
            else 0
        )
    else:
        min_thai = 0
    channel_cap = int(
        run_config.get("max_videos_per_channel_per_leaf")
        or (
            DEFAULT_MAX_VIDEOS_PER_CHANNEL_PER_LEAF
            if not candidate_count
            and run_config.get("source_video_duration_policy")
            == "unrestricted_positive_duration"
            else target
        )
    )
    return min_thai, max(1, channel_cap)


def _collection_progress(
    *,
    candidates: Sequence[dict[str, Any]],
    run_config: dict[str, Any],
    accepted_by_strategy: dict[str, Counter[str]] | None = None,
) -> dict[str, Any]:
    accepted_by_strategy = accepted_by_strategy or _accepted_counts(candidates)
    languages_by_leaf = _language_counts_by_leaf(candidates)
    _channel_counts, channels_by_leaf = _channel_counts_by_leaf(candidates)
    configured_leaves = [
        normalize_taxonomy_leaf(item)
        for item in (run_config.get("leaf_keys") or ())
    ]
    candidate_leaves = [
        normalize_taxonomy_leaf(item.get("proposed_leaf_key"))
        for item in candidates
    ]
    leaf_keys = list(
        dict.fromkeys(
            leaf_key
            for leaf_key in (*configured_leaves, *candidate_leaves)
            if leaf_key in ACTIVE_LEAF_KEYS
        )
    )
    target_per_leaf = max(1, int(run_config.get("target_per_leaf") or 1))
    sampling_plan = list(run_config.get("sampling_plan") or [])
    min_thai, channel_cap = _quality_values_for_progress(
        run_config,
        candidate_count=len(candidates),
    )
    configured_languages = [
        str(item).lower() for item in (run_config.get("languages") or ())
    ]
    total_languages: Counter[str] = Counter()
    leaf_items: list[dict[str, Any]] = []
    for leaf_key in leaf_keys:
        accepted = sum(
            counts[leaf_key] for counts in accepted_by_strategy.values()
        )
        language_counts = Counter(languages_by_leaf[leaf_key])
        for language in configured_languages:
            language_counts.setdefault(language, 0)
        total_languages.update(language_counts)
        thai_count = int(language_counts["th"])
        strategies_complete = all(
            accepted_by_strategy[str(plan["strategy"])][leaf_key]
            >= int(plan["target_per_leaf"])
            for plan in sampling_plan
        ) if sampling_plan else accepted >= target_per_leaf
        complete = strategies_complete and thai_count >= min_thai
        leaf_items.append(
            {
                "leaf_key": leaf_key,
                "target": target_per_leaf,
                "accepted": accepted,
                "remaining": max(0, target_per_leaf - accepted),
                "percent": round(
                    min(100.0, 100.0 * accepted / target_per_leaf),
                    1,
                ),
                "language_counts": dict(sorted(language_counts.items())),
                "thai_minimum": min_thai,
                "thai_remaining": max(0, min_thai - thai_count),
                "unique_channels": len(channels_by_leaf[leaf_key]),
                "minimum_unique_channels_expected": math.ceil(
                    target_per_leaf / channel_cap
                ),
                "max_videos_per_channel": channel_cap,
                "strategy_counts": {
                    strategy: int(counts[leaf_key])
                    for strategy, counts in accepted_by_strategy.items()
                },
                "complete": complete,
            }
        )

    target_total = target_per_leaf * len(leaf_keys)
    accepted_total = len(candidates)
    return {
        "target_total": target_total,
        "accepted_total": accepted_total,
        "remaining_total": max(0, target_total - accepted_total),
        "percent": round(
            min(100.0, 100.0 * accepted_total / max(target_total, 1)),
            1,
        ),
        "language_counts": dict(sorted(total_languages.items())),
        "unique_channels": len(
            {
                channel_id
                for channel_ids in channels_by_leaf.values()
                for channel_id in channel_ids
            }
        ),
        "complete_leaves": sum(1 for item in leaf_items if item["complete"]),
        "leaf_count": len(leaf_items),
        "by_leaf": leaf_items,
    }


def _all_targets_met(
    *,
    leaf_keys: Sequence[str],
    sampling_plan: Sequence[dict[str, Any]],
    accepted_by_strategy: dict[str, Counter[str]],
    language_counts_by_leaf: dict[str, Counter[str]],
    min_thai_per_leaf: int,
) -> bool:
    strategies_complete = all(
        accepted_by_strategy[str(plan["strategy"])][leaf_key]
        >= int(plan["target_per_leaf"])
        for plan in sampling_plan
        for leaf_key in leaf_keys
    )
    thai_complete = all(
        language_counts_by_leaf[leaf_key]["th"] >= min_thai_per_leaf
        for leaf_key in leaf_keys
    )
    return strategies_complete and thai_complete


def _persist_collection_state(
    db: Session,
    *,
    collection_run: DatasetCollectionRun,
    candidate_file: Path,
    review_file: Path,
    manifest_file: Path,
    run_config: dict[str, Any],
    candidates: list[dict[str, Any]],
    accepted_by_strategy: dict[str, Counter[str]],
    rejected_reasons: Counter[str],
    duplicate_reasons: Counter[str],
    quality_skip_reasons: Counter[str],
    candidates_seen: int,
    search_state: dict[str, Any],
    dedup_catalog_summary: dict[str, Any],
    status: str,
    failure_message: str | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    _annotate_performance_ranks(candidates)
    _write_text_atomic(
        candidate_file,
        "".join(_canonical_json(item) + "\n" for item in candidates),
    )
    _write_text_atomic(review_file, _review_csv_text(candidates))
    candidate_file_sha256 = _sha256_file(candidate_file)
    review_file_sha256 = _sha256_file(review_file)
    now = _utc_now()
    accepted_by_leaf: Counter[str] = Counter()
    for strategy_counts in accepted_by_strategy.values():
        accepted_by_leaf.update(strategy_counts)

    progress = _collection_progress(
        candidates=candidates,
        run_config=run_config,
        accepted_by_strategy=accepted_by_strategy,
    )
    manifest = {
        "schema_version": COLLECTION_SCHEMA_VERSION,
        "collection_run_id": collection_run.collection_run_id,
        "run_key": collection_run.run_key,
        "dataset_source": YOUTUBE_CC_DATASET_SOURCE,
        "dataset_version": collection_run.dataset_version,
        "taxonomy_version": TAXONOMY_VERSION,
        "status": status,
        "started_at": _iso_z(collection_run.started_at.replace(tzinfo=timezone.utc)),
        "updated_at": _iso_z(now),
        "completed_at": (
            None
            if status in {"running", "quota_waiting", "transcript_waiting"}
            else _iso_z(now)
        ),
        "resume_count": int(collection_run.resume_count or 0),
        "config": run_config,
        "candidate_artifact": {
            "path": str(candidate_file),
            "sha256": candidate_file_sha256,
            "rows": len(candidates),
        },
        "human_review_template": {
            "path": str(review_file),
            "sha256": review_file_sha256,
            "auto_approved_rows": 0,
        },
        "accepted_by_leaf": dict(accepted_by_leaf),
        "accepted_by_strategy": {
            strategy: dict(counts)
            for strategy, counts in accepted_by_strategy.items()
        },
        "progress": progress,
        "rejected_reasons": dict(rejected_reasons),
        "quality_filters": {
            "skipped_total": sum(quality_skip_reasons.values()),
            "skipped_by_reason": dict(quality_skip_reasons),
        },
        "deduplication": {
            **dedup_catalog_summary,
            "skipped_total": sum(duplicate_reasons.values()),
            "skipped_by_reason": dict(duplicate_reasons),
        },
        "search_state": search_state,
    }
    if failure_message:
        manifest["failure"] = failure_message[:2000]
    if status == "quota_waiting":
        manifest["quota"] = {
            "retryable": True,
            "message": "YouTube search quota is exhausted; resume this run after reset.",
        }
    if status == "transcript_waiting":
        manifest["transcript_provider"] = {
            "retryable": True,
            "message": (
                "The public transcript provider blocked this IP. Resume this run "
                "after the block clears or from a permitted network."
            ),
        }
    _write_text_atomic(
        manifest_file,
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    )

    collection_run.status = status
    collection_run.candidate_artifact_path = str(candidate_file)
    collection_run.candidate_artifact_sha256 = candidate_file_sha256
    collection_run.review_artifact_path = str(review_file)
    collection_run.review_artifact_sha256 = review_file_sha256
    collection_run.manifest_path = str(manifest_file)
    collection_run.manifest_sha256 = _sha256_file(manifest_file)
    collection_run.candidates_seen = candidates_seen
    collection_run.transcripts_collected = len(candidates)
    collection_run.duplicates_skipped = sum(duplicate_reasons.values())
    collection_run.errors_count = sum(rejected_reasons.values()) + int(
        bool(failure_message)
        and status not in {"quota_waiting", "transcript_waiting"}
    )
    collection_run.completed_at = (
        None
        if status in {"running", "quota_waiting", "transcript_waiting"}
        else now.replace(tzinfo=None)
    )
    db.commit()
    if progress_callback is not None:
        progress_callback(manifest)
    return manifest


def _execute_collection(
    db: Session,
    *,
    collection_run: DatasetCollectionRun,
    run_config: dict[str, Any],
    candidate_file: Path,
    review_file: Path,
    manifest_file: Path,
    candidates: list[dict[str, Any]],
    search_state: dict[str, Any],
    rejected_reasons: Counter[str],
    duplicate_reasons: Counter[str],
    quality_skip_reasons: Counter[str],
    candidates_seen: int,
    api_key: str,
    page_budget_per_query: int,
    timeout_seconds: float,
    transcript_fetcher: Callable[[str, Sequence[str]], dict[str, Any]],
    youtube_getter: Callable[..., dict[str, Any]],
    progress_callback: Callable[[dict[str, Any]], None] | None,
) -> dict[str, Any]:
    leaf_keys = tuple(run_config["leaf_keys"])
    languages = tuple(run_config["languages"])
    region_code = str(run_config["region_code"])
    sampling_plan = list(run_config["sampling_plan"])
    accepted_by_strategy = _accepted_counts(candidates)
    accepted_by_language = _language_counts_by_leaf(candidates)
    current_channel_counts, _current_channels = _channel_counts_by_leaf(candidates)
    min_thai_per_leaf = int(run_config.get("min_thai_per_leaf") or 0)
    max_videos_per_channel = int(
        run_config.get("max_videos_per_channel_per_leaf")
        or run_config["target_per_leaf"]
    )
    max_non_thai_per_leaf = int(run_config["target_per_leaf"]) - min_thai_per_leaf
    candidate_video_ids = {
        str(item.get("source_youtube_id") or "").strip()
        for item in candidates
    }
    candidate_transcript_hashes = {
        str(item.get("transcript_sha256") or "").strip()
        for item in candidates
    }
    attempted_video_ids = set(candidate_video_ids)
    (
        catalog_video_ids,
        catalog_transcript_hashes,
        catalog_channel_counts,
        catalog_summary,
    ) = _dedup_catalog(db, exclude_run_id=collection_run.collection_run_id)

    def target_met(strategy: str, leaf_key: str, target: int) -> bool:
        return accepted_by_strategy[strategy][leaf_key] >= target

    try:
        _persist_collection_state(
            db,
            collection_run=collection_run,
            candidate_file=candidate_file,
            review_file=review_file,
            manifest_file=manifest_file,
            run_config=run_config,
            candidates=candidates,
            accepted_by_strategy=accepted_by_strategy,
            rejected_reasons=rejected_reasons,
            duplicate_reasons=duplicate_reasons,
            quality_skip_reasons=quality_skip_reasons,
            candidates_seen=candidates_seen,
            search_state=search_state,
            dedup_catalog_summary=catalog_summary,
            status="running",
            progress_callback=progress_callback,
        )
        for plan in sampling_plan:
            strategy = str(plan["strategy"])
            search_order = str(plan["search_order"])
            strategy_target = int(plan["target_per_leaf"])
            for leaf_key in leaf_keys:
                if target_met(strategy, leaf_key, strategy_target):
                    continue
                configured_queries = (
                    (run_config.get("queries_by_leaf") or {}).get(leaf_key)
                    or collection_queries_for_leaf(leaf_key)
                )
                for query in configured_queries:
                    if target_met(strategy, leaf_key, strategy_target):
                        break
                    state_key = _sha256_text(f"{strategy}|{leaf_key}|{query}")
                    query_state = dict(search_state.get(state_key) or {})
                    if query_state.get("exhausted"):
                        continue
                    page_token = str(query_state.get("next_page_token") or "").strip() or None
                    query_language = (
                        "th" if re.search(r"[\u0E00-\u0E7F]", query) else "en"
                    )
                    for _page in range(page_budget_per_query):
                        search_params: dict[str, Any] = {
                            "part": "snippet",
                            "q": query,
                            "type": "video",
                            "videoLicense": "creativeCommon",
                            "videoCaption": "closedCaption",
                            "maxResults": 50,
                            "order": search_order,
                            "regionCode": region_code,
                            "relevanceLanguage": query_language,
                            "safeSearch": "moderate",
                        }
                        if page_token:
                            search_params["pageToken"] = page_token
                        search = youtube_getter(
                            "search",
                            api_key=api_key,
                            timeout_seconds=timeout_seconds,
                            **search_params,
                        )
                        raw_video_ids = [
                            str((item.get("id") or {}).get("videoId") or "").strip()
                            for item in search.get("items") or []
                        ]
                        video_ids: list[str] = []
                        page_video_ids: set[str] = set()
                        for video_id in raw_video_ids:
                            if not video_id:
                                continue
                            if video_id in candidate_video_ids:
                                duplicate_reasons["duplicate_in_run_video"] += 1
                                continue
                            if video_id in catalog_video_ids:
                                duplicate_reasons["duplicate_previous_video"] += 1
                                continue
                            if video_id in attempted_video_ids:
                                duplicate_reasons["duplicate_search_result"] += 1
                                continue
                            if video_id in page_video_ids:
                                duplicate_reasons["duplicate_search_result"] += 1
                                continue
                            page_video_ids.add(video_id)
                            video_ids.append(video_id)

                        if video_ids:
                            details = youtube_getter(
                                "videos",
                                api_key=api_key,
                                timeout_seconds=timeout_seconds,
                                part="snippet,contentDetails,statistics,status",
                                id=",".join(video_ids),
                                maxResults=50,
                            )
                            for search_rank, item in enumerate(
                                details.get("items") or [],
                                start=1,
                            ):
                                candidates_seen += 1
                                video_id = str(item.get("id") or "").strip()
                                attempted_video_ids.add(video_id)
                                status = item.get("status") or {}
                                content_details = item.get("contentDetails") or {}
                                snippet = item.get("snippet") or {}
                                try:
                                    duration = parse_iso8601_duration(
                                        str(content_details.get("duration") or "")
                                    )
                                except YouTubeCCDatasetError:
                                    rejected_reasons["invalid_duration"] += 1
                                    continue
                                if status.get("license") != "creativeCommon":
                                    rejected_reasons["not_creative_common"] += 1
                                    continue
                                if duration <= 0:
                                    rejected_reasons["non_positive_duration"] += 1
                                    continue
                                if str(content_details.get("caption") or "").lower() != "true":
                                    rejected_reasons["no_caption_flag"] += 1
                                    continue
                                if str(snippet.get("liveBroadcastContent") or "none") != "none":
                                    rejected_reasons["live_broadcast"] += 1
                                    continue
                                channel_id = str(snippet.get("channelId") or "").strip()
                                if not channel_id:
                                    rejected_reasons["missing_channel_id"] += 1
                                    continue
                                if (
                                    catalog_channel_counts[(leaf_key, channel_id)]
                                    + current_channel_counts[(leaf_key, channel_id)]
                                    >= max_videos_per_channel
                                ):
                                    quality_skip_reasons["channel_cap_reached"] += 1
                                    continue
                                try:
                                    transcript = transcript_fetcher(video_id, languages)
                                except YouTubeTranscriptProviderBlockedError:
                                    rejected_reasons["transcript_provider_blocked"] += 1
                                    raise
                                except Exception:
                                    rejected_reasons["transcript_unavailable"] += 1
                                    continue
                                transcript_language = str(
                                    transcript.get("language") or "und"
                                ).lower().split("-", 1)[0]
                                if transcript_language not in languages:
                                    rejected_reasons["unsupported_transcript_language"] += 1
                                    continue
                                current_non_thai = sum(
                                    count
                                    for language, count in accepted_by_language[
                                        leaf_key
                                    ].items()
                                    if language != "th"
                                )
                                if (
                                    transcript_language != "th"
                                    and current_non_thai >= max_non_thai_per_leaf
                                ):
                                    quality_skip_reasons[
                                        "non_thai_capacity_reserved"
                                    ] += 1
                                    continue
                                transcript_hash = str(
                                    transcript.get("transcript_sha256") or ""
                                ).strip()
                                if not transcript_hash:
                                    rejected_reasons["missing_transcript_hash"] += 1
                                    continue
                                if transcript_hash in candidate_transcript_hashes:
                                    duplicate_reasons["duplicate_in_run_transcript"] += 1
                                    continue
                                if transcript_hash in catalog_transcript_hashes:
                                    duplicate_reasons["duplicate_previous_transcript"] += 1
                                    continue
                                candidate = _make_candidate(
                                    item=item,
                                    transcript=transcript,
                                    run_key=collection_run.run_key,
                                    dataset_version=collection_run.dataset_version,
                                    leaf_key=leaf_key,
                                    query=query,
                                    region_code=region_code,
                                    collected_at=_utc_now(),
                                    collection_strategy=strategy,
                                    search_order=search_order,
                                    search_rank=(
                                        int(query_state.get("pages_fetched") or 0) * 50
                                        + search_rank
                                    ),
                                )
                                candidates.append(candidate)
                                candidate_video_ids.add(video_id)
                                candidate_transcript_hashes.add(transcript_hash)
                                accepted_by_strategy[strategy][leaf_key] += 1
                                accepted_by_language[leaf_key][transcript_language] += 1
                                current_channel_counts[(leaf_key, channel_id)] += 1
                                if target_met(strategy, leaf_key, strategy_target):
                                    break

                        next_page_token = (
                            str(search.get("nextPageToken") or "").strip() or None
                        )
                        query_state = {
                            "strategy": strategy,
                            "leaf_key": leaf_key,
                            "query": query,
                            "search_order": search_order,
                            "pages_fetched": int(query_state.get("pages_fetched") or 0) + 1,
                            "next_page_token": next_page_token,
                            "exhausted": next_page_token is None,
                            "updated_at": _iso_z(_utc_now()),
                        }
                        search_state[state_key] = query_state
                        _persist_collection_state(
                            db,
                            collection_run=collection_run,
                            candidate_file=candidate_file,
                            review_file=review_file,
                            manifest_file=manifest_file,
                            run_config=run_config,
                            candidates=candidates,
                            accepted_by_strategy=accepted_by_strategy,
                            rejected_reasons=rejected_reasons,
                            duplicate_reasons=duplicate_reasons,
                            quality_skip_reasons=quality_skip_reasons,
                            candidates_seen=candidates_seen,
                            search_state=search_state,
                            dedup_catalog_summary=catalog_summary,
                            status="running",
                            progress_callback=progress_callback,
                        )
                        if target_met(strategy, leaf_key, strategy_target):
                            break
                        page_token = next_page_token
                        if not page_token:
                            break

        final_status = (
            "collected"
            if _all_targets_met(
                leaf_keys=leaf_keys,
                sampling_plan=sampling_plan,
                accepted_by_strategy=accepted_by_strategy,
                language_counts_by_leaf=accepted_by_language,
                min_thai_per_leaf=min_thai_per_leaf,
            )
            else "partial"
        )
        return _persist_collection_state(
            db,
            collection_run=collection_run,
            candidate_file=candidate_file,
            review_file=review_file,
            manifest_file=manifest_file,
            run_config=run_config,
            candidates=candidates,
            accepted_by_strategy=accepted_by_strategy,
            rejected_reasons=rejected_reasons,
            duplicate_reasons=duplicate_reasons,
            quality_skip_reasons=quality_skip_reasons,
            candidates_seen=candidates_seen,
            search_state=search_state,
            dedup_catalog_summary=catalog_summary,
            status=final_status,
            progress_callback=progress_callback,
        )
    except BaseException as exc:
        db.rollback()
        failed_run = db.get(DatasetCollectionRun, collection_run.collection_run_id)
        if failed_run is not None:
            try:
                quota_waiting = isinstance(exc, YouTubeQuotaExceededError)
                transcript_waiting = isinstance(
                    exc,
                    YouTubeTranscriptProviderBlockedError,
                )
                waiting_status = (
                    "quota_waiting"
                    if quota_waiting
                    else "transcript_waiting"
                    if transcript_waiting
                    else "failed"
                )
                _persist_collection_state(
                    db,
                    collection_run=failed_run,
                    candidate_file=candidate_file,
                    review_file=review_file,
                    manifest_file=manifest_file,
                    run_config=run_config,
                    candidates=candidates,
                    accepted_by_strategy=accepted_by_strategy,
                    rejected_reasons=rejected_reasons,
                    duplicate_reasons=duplicate_reasons,
                    quality_skip_reasons=quality_skip_reasons,
                    candidates_seen=candidates_seen,
                    search_state=search_state,
                    dedup_catalog_summary=catalog_summary,
                    status=waiting_status,
                    failure_message=f"{type(exc).__name__}: {exc}",
                    progress_callback=progress_callback,
                )
            except Exception:
                db.rollback()
        raise


def collect_youtube_cc_candidates(
    db: Session,
    *,
    api_key: str,
    candidate_path: str | Path,
    review_path: str | Path,
    manifest_path: str | Path,
    dataset_version: str = DEFAULT_YOUTUBE_CC_DATASET_VERSION,
    leaf_keys: Sequence[str] = ACTIVE_LEAF_KEYS,
    target_per_leaf: int = 50,
    performance_target_per_leaf: int | None = None,
    languages: Sequence[str] = SUPPORTED_TRANSCRIPT_LANGUAGES,
    min_thai_per_leaf: int | None = None,
    max_videos_per_channel_per_leaf: int = (
        DEFAULT_MAX_VIDEOS_PER_CHANNEL_PER_LEAF
    ),
    region_code: str = "TH",
    max_pages_per_query: int = 2,
    timeout_seconds: float = 10.0,
    transcript_fetcher: Callable[[str, Sequence[str]], dict[str, Any]] = fetch_public_transcript,
    youtube_getter: Callable[..., dict[str, Any]] = _youtube_get,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    normalized_leaves, normalized_languages = _normalize_collection_inputs(
        leaf_keys=leaf_keys,
        languages=languages,
        max_pages_per_query=max_pages_per_query,
    )
    sampling_plan = _sampling_plan(target_per_leaf, performance_target_per_leaf)
    normalized_min_thai, normalized_channel_cap = _quality_collection_settings(
        target_per_leaf=target_per_leaf,
        languages=normalized_languages,
        min_thai_per_leaf=min_thai_per_leaf,
        max_videos_per_channel_per_leaf=max_videos_per_channel_per_leaf,
    )
    started_at = _utc_now()
    candidate_file = _resolve_project_path(candidate_path)
    review_file = _resolve_project_path(review_path)
    manifest_file = _resolve_project_path(manifest_path)
    existing_artifacts = [
        str(path)
        for path in (candidate_file, review_file, manifest_file)
        if path.exists()
    ]
    if existing_artifacts:
        raise YouTubeCCDatasetError(
            "New collection artifacts already exist: " + ", ".join(existing_artifacts)
        )
    run_config = {
        "schema_version": COLLECTION_SCHEMA_VERSION,
        "dataset_version": dataset_version,
        "leaf_keys": normalized_leaves,
        "target_per_leaf": target_per_leaf,
        "performance_target_per_leaf": next(
            (
                int(plan["target_per_leaf"])
                for plan in sampling_plan
                if plan["strategy"] == RECOMMENDATION_HIGH_PERFORMANCE_STRATEGY
            ),
            0,
        ),
        "sampling_plan": sampling_plan,
        "min_thai_per_leaf": normalized_min_thai,
        "max_videos_per_channel_per_leaf": normalized_channel_cap,
        "language_balance_policy": "reserve_minimum_thai_v1",
        "channel_diversity_policy": "max_per_channel_per_leaf_v1",
        "queries_by_leaf": _queries_by_leaf(normalized_leaves),
        "source_video_duration_policy": "unrestricted_positive_duration",
        "transcript_window_seconds": TRANSCRIPT_WINDOW_SECONDS,
        "duration_recommendation_max_seconds": RECOMMENDATION_DURATION_MAX_SECONDS,
        "languages": normalized_languages,
        "region_code": region_code.upper(),
        "max_pages_per_query": max_pages_per_query,
        "started_at": _iso_z(started_at),
    }
    run_key = _sha256_text(_canonical_json(run_config))
    collection_run = DatasetCollectionRun(
        run_key=run_key,
        dataset_source=YOUTUBE_CC_DATASET_SOURCE,
        dataset_version=dataset_version,
        status="running",
        region_code=region_code.upper(),
        languages_json=json.dumps(normalized_languages, ensure_ascii=False),
        query_config_json=json.dumps(run_config, ensure_ascii=False, sort_keys=True),
        candidate_artifact_path=str(candidate_file),
        review_artifact_path=str(review_file),
        manifest_path=str(manifest_file),
        started_at=started_at.replace(tzinfo=None),
    )
    db.add(collection_run)
    db.commit()
    db.refresh(collection_run)
    return _execute_collection(
        db,
        collection_run=collection_run,
        run_config=run_config,
        candidate_file=candidate_file,
        review_file=review_file,
        manifest_file=manifest_file,
        candidates=[],
        search_state={},
        rejected_reasons=Counter(),
        duplicate_reasons=Counter(),
        quality_skip_reasons=Counter(),
        candidates_seen=0,
        api_key=api_key,
        page_budget_per_query=max_pages_per_query,
        timeout_seconds=timeout_seconds,
        transcript_fetcher=transcript_fetcher,
        youtube_getter=youtube_getter,
        progress_callback=progress_callback,
    )


def resume_youtube_cc_collection(
    db: Session,
    *,
    collection_run_id: int,
    api_key: str,
    max_pages_per_query: int | None = None,
    timeout_seconds: float = 10.0,
    transcript_fetcher: Callable[[str, Sequence[str]], dict[str, Any]] = fetch_public_transcript,
    youtube_getter: Callable[..., dict[str, Any]] = _youtube_get,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    collection_run = db.get(DatasetCollectionRun, collection_run_id)
    if collection_run is None:
        raise YouTubeCCDatasetError(f"Collection run {collection_run_id} not found")
    if collection_run.dataset_source != YOUTUBE_CC_DATASET_SOURCE:
        raise YouTubeCCDatasetError("Only YouTube CC collection runs can be resumed")
    review_event_count = (
        db.query(DatasetReviewEvent)
        .filter(DatasetReviewEvent.collection_run_id == collection_run_id)
        .count()
    )
    if review_event_count:
        raise YouTubeCCDatasetError(
            "This run already has human review events and is immutable; start a new run instead"
        )

    try:
        run_config = json.loads(collection_run.query_config_json)
    except (TypeError, json.JSONDecodeError) as exc:
        raise YouTubeCCDatasetError("Collection run has invalid query configuration") from exc
    required_config = {
        "leaf_keys",
        "languages",
        "region_code",
        "target_per_leaf",
        "max_pages_per_query",
    }
    if not required_config.issubset(run_config):
        raise YouTubeCCDatasetError(
            "This legacy/reconstructed run has no resumable search configuration"
        )
    if run_config.get("source_video_duration_policy") != (
        "unrestricted_positive_duration"
    ):
        raise YouTubeCCDatasetError(
            "This run used the legacy five-minute dataset filter and cannot be resumed "
            "without missing long videos from previously scanned pages; start a new run"
        )
    if "sampling_plan" not in run_config:
        run_config["sampling_plan"] = _sampling_plan(
            int(run_config["target_per_leaf"]),
            0,
        )
    run_config = _upgrade_empty_run_quality_config(collection_run, run_config)
    _quality_collection_settings(
        target_per_leaf=int(run_config["target_per_leaf"]),
        languages=run_config["languages"],
        min_thai_per_leaf=int(run_config["min_thai_per_leaf"]),
        max_videos_per_channel_per_leaf=int(
            run_config["max_videos_per_channel_per_leaf"]
        ),
    )
    _normalize_collection_inputs(
        leaf_keys=run_config["leaf_keys"],
        languages=run_config["languages"],
        max_pages_per_query=int(run_config["max_pages_per_query"]),
    )
    page_budget = (
        int(max_pages_per_query)
        if max_pages_per_query is not None
        else int(run_config["max_pages_per_query"])
    )
    if page_budget < 1:
        raise YouTubeCCDatasetError("max_pages_per_query must be at least 1")

    manifest = _load_collection_manifest(collection_run, required=False)
    raw_candidate_path = str(collection_run.candidate_artifact_path or "").strip()
    if not raw_candidate_path:
        raise YouTubeCCDatasetError("Collection run has no candidate artifact path")
    candidate_file = _resolve_project_path(raw_candidate_path)
    if candidate_file.is_file():
        actual_hash = _sha256_file(candidate_file)
        if (
            collection_run.candidate_artifact_sha256
            and collection_run.candidate_artifact_sha256 != actual_hash
        ):
            raise YouTubeCCDatasetError("Candidate artifact hash differs from collection run")
        candidates = list(
            _load_candidates(candidate_file, allow_empty=True).values()
        )
    else:
        candidates = []

    raw_review_path = str(collection_run.review_artifact_path or "").strip()
    if not raw_review_path:
        raw_review_path = str(
            (manifest.get("human_review_template") or {}).get("path") or ""
        ).strip()
    if not raw_review_path:
        raise YouTubeCCDatasetError("Collection run has no review artifact path")
    review_file = _resolve_project_path(raw_review_path)
    if review_file.is_file():
        actual_review_hash = _sha256_file(review_file)
        if (
            collection_run.review_artifact_sha256
            and collection_run.review_artifact_sha256 != actual_review_hash
        ):
            raise YouTubeCCDatasetError("Review artifact hash differs from collection run")
        if any(
            str(row.get("decision") or "").strip()
            for row in _review_rows(review_file)
        ):
            raise YouTubeCCDatasetError(
                "The review artifact already contains decisions and cannot be regenerated"
            )
    raw_manifest_path = str(collection_run.manifest_path or "").strip()
    if not raw_manifest_path:
        raise YouTubeCCDatasetError("Collection run has no manifest path")
    manifest_file = _resolve_project_path(raw_manifest_path)

    accepted_by_strategy = _accepted_counts(candidates)
    accepted_by_language = _language_counts_by_leaf(candidates)
    if _all_targets_met(
        leaf_keys=run_config["leaf_keys"],
        sampling_plan=run_config["sampling_plan"],
        accepted_by_strategy=accepted_by_strategy,
        language_counts_by_leaf=accepted_by_language,
        min_thai_per_leaf=int(run_config["min_thai_per_leaf"]),
    ):
        raise YouTubeCCDatasetError("Collection run already meets every sampling target")

    collection_run.status = "running"
    collection_run.resume_count = int(collection_run.resume_count or 0) + 1
    collection_run.last_resumed_at = _utc_now().replace(tzinfo=None)
    collection_run.completed_at = None
    collection_run.review_artifact_path = str(review_file)
    collection_run.query_config_json = json.dumps(
        run_config,
        ensure_ascii=False,
        sort_keys=True,
    )
    db.commit()
    return _execute_collection(
        db,
        collection_run=collection_run,
        run_config=run_config,
        candidate_file=candidate_file,
        review_file=review_file,
        manifest_file=manifest_file,
        candidates=candidates,
        search_state=dict(manifest.get("search_state") or {}),
        rejected_reasons=Counter(manifest.get("rejected_reasons") or {}),
        duplicate_reasons=Counter(
            (manifest.get("deduplication") or {}).get("skipped_by_reason") or {}
        ),
        quality_skip_reasons=Counter(
            (manifest.get("quality_filters") or {}).get("skipped_by_reason") or {}
        ),
        candidates_seen=int(collection_run.candidates_seen or 0),
        api_key=api_key,
        page_budget_per_query=page_budget,
        timeout_seconds=timeout_seconds,
        transcript_fetcher=transcript_fetcher,
        youtube_getter=youtube_getter,
        progress_callback=progress_callback,
    )


def _load_candidates(
    path: Path,
    *,
    allow_empty: bool = False,
) -> dict[str, dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError as exc:
            raise YouTubeCCDatasetError(
                f"Invalid candidate JSONL at line {line_number}: {exc}"
            ) from exc
        video_id = str(candidate.get("source_youtube_id") or "").strip()
        if not video_id:
            raise YouTubeCCDatasetError(f"Candidate line {line_number} has no YouTube ID")
        expected_hash = _candidate_hash(candidate)
        if candidate.get("candidate_sha256") != expected_hash:
            raise YouTubeCCDatasetError(
                f"Candidate hash mismatch for YouTube ID {video_id}"
            )
        if video_id in candidates:
            raise YouTubeCCDatasetError(f"Duplicate candidate YouTube ID: {video_id}")
        candidates[video_id] = candidate
    if not candidates and not allow_empty:
        raise YouTubeCCDatasetError(f"Candidate artifact is empty: {path}")
    return candidates


def _split_for_channel(channel_id: str) -> tuple[str, str]:
    return channel_dataset_split(channel_id)


def _performance_signal(candidate: dict[str, Any]) -> float:
    raw_signal = candidate.get("performance_signal")
    if raw_signal is not None:
        return round(float(raw_signal), 6)
    return _performance_metrics(candidate)["performance_signal"]


def _review_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [field for field in REVIEW_FIELDS if field not in (reader.fieldnames or [])]
        if missing:
            raise YouTubeCCDatasetError(
                "Review CSV is missing columns: " + ", ".join(missing)
            )
        return [dict(row) for row in reader]


def import_reviewed_youtube_cc_dataset(
    db: Session,
    *,
    candidate_path: str | Path,
    review_path: str | Path,
) -> dict[str, Any]:
    candidate_file = Path(candidate_path)
    review_file = Path(review_path)
    if not candidate_file.is_file():
        raise YouTubeCCDatasetError(f"Candidate artifact not found: {candidate_file}")
    if not review_file.is_file():
        raise YouTubeCCDatasetError(f"Review CSV not found: {review_file}")

    candidates = _load_candidates(candidate_file)
    reviews = _review_rows(review_file)
    candidate_file_sha256 = _sha256_file(candidate_file)
    review_file_sha256 = _sha256_file(review_file)
    first_candidate = next(iter(candidates.values()))
    run_key = str(first_candidate.get("run_key") or "")
    dataset_version = str(first_candidate.get("dataset_version") or "")
    if not run_key or not dataset_version:
        raise YouTubeCCDatasetError("Candidate artifact has no run or dataset version")

    collection_run = (
        db.query(DatasetCollectionRun)
        .filter(DatasetCollectionRun.run_key == run_key)
        .first()
    )
    if collection_run is None:
        collection_run = DatasetCollectionRun(
            run_key=run_key,
            dataset_source=YOUTUBE_CC_DATASET_SOURCE,
            dataset_version=dataset_version,
            status="collected",
            region_code=str(first_candidate.get("region_code") or "TH"),
            languages_json=json.dumps(
                sorted({str(item.get("transcript_language")) for item in candidates.values()}),
                ensure_ascii=False,
            ),
            query_config_json=json.dumps(
                {"reconstructed_from_artifact": str(candidate_file)},
                ensure_ascii=False,
            ),
            candidate_artifact_path=str(candidate_file),
            candidate_artifact_sha256=candidate_file_sha256,
            candidates_seen=len(candidates),
            transcripts_collected=len(candidates),
            completed_at=_utc_now().replace(tzinfo=None),
        )
        db.add(collection_run)
        db.flush()
    elif (
        collection_run.candidate_artifact_sha256
        and collection_run.candidate_artifact_sha256 != candidate_file_sha256
    ):
        raise YouTubeCCDatasetError("Candidate artifact hash differs from collection run")

    import_batch_id = _sha256_text(
        "|".join(
            (
                candidate_file_sha256,
                review_file_sha256,
                dataset_version,
                TAXONOMY_VERSION,
            )
        )
    )
    stats: Counter[str] = Counter()
    errors: list[str] = []

    for row_number, review in enumerate(reviews, start=2):
        decision = str(review.get("decision") or "").strip().lower()
        if not decision:
            stats["pending"] += 1
            continue
        if decision not in {"approve", "reject"}:
            errors.append(f"row {row_number}: decision must be approve or reject")
            continue
        video_id = str(review.get("source_youtube_id") or "").strip()
        candidate = candidates.get(video_id)
        if candidate is None:
            errors.append(f"row {row_number}: unknown YouTube ID {video_id}")
            continue
        if review.get("candidate_sha256") != candidate.get("candidate_sha256"):
            errors.append(f"row {row_number}: candidate hash mismatch for {video_id}")
            continue
        reviewer = str(review.get("reviewer") or "").strip()
        if not reviewer:
            errors.append(f"row {row_number}: reviewer is required")
            continue
        try:
            reviewed_at = _parse_datetime(review.get("reviewed_at"), field="reviewed_at")
        except YouTubeCCDatasetError as exc:
            errors.append(f"row {row_number}: {exc}")
            continue

        proposed_leaf = normalize_taxonomy_leaf(candidate.get("proposed_leaf_key"))
        reviewed_leaf = normalize_taxonomy_leaf(review.get("reviewed_leaf_key"))
        quality = str(review.get("transcript_quality") or "").strip().lower()
        notes = str(review.get("review_notes") or "").strip() or None

        if decision == "approve":
            if reviewed_leaf not in ACTIVE_LEAF_KEYS:
                errors.append(f"row {row_number}: reviewed_leaf_key is required")
                continue
            if quality not in ACCEPTED_TRANSCRIPT_QUALITIES:
                errors.append(
                    f"row {row_number}: transcript_quality must be one of "
                    + ", ".join(ACCEPTED_TRANSCRIPT_QUALITIES)
                )
                continue
            if candidate.get("youtube_license_code") != "creativeCommon":
                errors.append(f"row {row_number}: license is not Creative Commons")
                continue
            if candidate.get("transcript_source") != YOUTUBE_CC_TRANSCRIPT_SOURCE:
                errors.append(f"row {row_number}: transcript source is not accepted")
                continue
            if candidate.get("caption_type") not in SUPPORTED_CAPTION_TYPES:
                errors.append(f"row {row_number}: caption type is not accepted")
                continue
            if candidate.get("transcript_language") not in SUPPORTED_TRANSCRIPT_LANGUAGES:
                errors.append(f"row {row_number}: transcript language is not accepted")
                continue

        existing_event = (
            db.query(DatasetReviewEvent)
            .filter(
                DatasetReviewEvent.collection_run_id == collection_run.collection_run_id,
                DatasetReviewEvent.source_youtube_id == video_id,
                DatasetReviewEvent.review_artifact_sha256 == review_file_sha256,
                DatasetReviewEvent.decision == decision,
            )
            .first()
        )
        if existing_event is not None:
            stats["unchanged"] += 1
            continue

        dataset = (
            db.query(DatasetContent)
            .filter(DatasetContent.source_youtube_id == video_id)
            .first()
        )
        if decision == "reject":
            if dataset is not None and dataset.dataset_source == YOUTUBE_CC_DATASET_SOURCE:
                dataset.is_active = False
                dataset.is_training_eligible = False
            db.add(
                DatasetReviewEvent(
                    collection_run_id=collection_run.collection_run_id,
                    dataset_id=dataset.dataset_id if dataset else None,
                    source_youtube_id=video_id,
                    decision=decision,
                    proposed_leaf_key=proposed_leaf,
                    reviewed_leaf_key=None,
                    transcript_quality=quality or None,
                    reviewer=reviewer,
                    notes=notes,
                    review_artifact_sha256=review_file_sha256,
                    reviewed_at=reviewed_at,
                )
            )
            stats["rejected"] += 1
            continue

        transcript_hash = str(candidate.get("transcript_sha256") or "")
        duplicate_transcript = (
            db.query(DatasetContent)
            .filter(
                DatasetContent.transcript_sha256 == transcript_hash,
                DatasetContent.source_youtube_id != video_id,
            )
            .first()
        )
        if duplicate_transcript is not None:
            errors.append(
                f"row {row_number}: transcript duplicates dataset {duplicate_transcript.dataset_id}"
            )
            continue
        if dataset is not None and dataset.dataset_source != YOUTUBE_CC_DATASET_SOURCE:
            errors.append(
                f"row {row_number}: YouTube ID already belongs to {dataset.dataset_source}"
            )
            continue

        split, creator_group_key = _split_for_channel(str(candidate.get("channel_id") or ""))
        path = taxonomy_path(reviewed_leaf)
        raw_metadata = dict(candidate.get("raw_metadata") or {})
        raw_metadata["collection"] = {
            "strategy": str(candidate.get("collection_strategy") or ""),
            "search_order": str(candidate.get("search_order") or ""),
            "search_rank": int(candidate.get("search_rank") or 0),
            "performance_rank_within_leaf": int(
                candidate.get("performance_rank_within_leaf") or 0
            ),
            "performance_percentile_within_leaf": float(
                candidate.get("performance_percentile_within_leaf") or 0.0
            ),
        }
        performance_metrics = _performance_metrics(candidate)
        values = {
            "title": str(candidate.get("title") or video_id)[:255],
            "video_url": str(candidate.get("video_url") or ""),
            "transcript": str(candidate.get("transcript") or ""),
            "category": reviewed_leaf,
            "source_platform": "youtube",
            "dataset_source": YOUTUBE_CC_DATASET_SOURCE,
            "dataset_version": dataset_version,
            "collection_run_id": collection_run.collection_run_id,
            "source_record_id": video_id,
            "source_youtube_id": video_id,
            "source_creator": str(candidate.get("channel_title") or ""),
            "source_channel_id": str(candidate.get("channel_id") or ""),
            "source_category": str(candidate.get("youtube_category_id") or "uncategorized"),
            "source_subcategory": proposed_leaf,
            "collection_query": str(candidate.get("collection_query") or ""),
            "source_release_url": str(candidate.get("video_url") or ""),
            "source_archive_sha256": candidate_file_sha256,
            "source_annotation_path": str(review_file),
            "source_annotation_sha256": review_file_sha256,
            "import_batch_id": import_batch_id,
            "taxonomy_version": TAXONOMY_VERSION,
            "taxonomy_leaf_key": reviewed_leaf,
            "category_level_1": path["category_level_1"],
            "category_level_2": path["category_level_2"],
            "category_level_3": path["category_level_3"],
            "language": str(candidate.get("transcript_language") or ""),
            "verification_status": YOUTUBE_CC_VERIFICATION_STATUS,
            "label_source": YOUTUBE_CC_LABEL_SOURCE,
            "license_name": YOUTUBE_CC_LICENSE_NAME,
            "license_url": YOUTUBE_CC_LICENSE_URL,
            "data_split": split,
            "split_strategy": SPLIT_STRATEGY,
            "creator_group_key": creator_group_key,
            "transcript_sha256": transcript_hash,
            "transcript_segment_count": int(candidate.get("transcript_segment_count") or 0),
            "transcript_start_seconds": float(candidate.get("transcript_start_seconds") or 0),
            "transcript_end_seconds": float(candidate.get("transcript_end_seconds") or 0),
            "transcript_window_seconds": TRANSCRIPT_WINDOW_SECONDS,
            "transcript_source": YOUTUBE_CC_TRANSCRIPT_SOURCE,
            "caption_type": str(candidate.get("caption_type") or ""),
            "transcript_quality": quality,
            "reviewed_by": reviewer,
            "reviewed_at": reviewed_at,
            "review_notes": notes,
            "statistics_captured_at": _parse_datetime(
                candidate.get("statistics_captured_at"),
                field="statistics_captured_at",
            ),
            "license_verified_at": _parse_datetime(
                candidate.get("license_verified_at"),
                field="license_verified_at",
            ),
            "raw_metadata_json": json.dumps(
                raw_metadata,
                ensure_ascii=False,
                sort_keys=True,
            ),
            "collection_strategy": str(
                candidate.get("collection_strategy")
                or CLASSIFICATION_DIVERSE_STRATEGY
            ),
            "average_views_per_day": performance_metrics["average_views_per_day"],
            "engagement_rate": performance_metrics["engagement_rate"],
            "is_training_eligible": True,
            "is_keyword_recommendation_eligible": True,
            "is_duration_recommendation_eligible": (
                0
                < int(candidate.get("duration_seconds") or 0)
                <= RECOMMENDATION_DURATION_MAX_SECONDS
            ),
            "is_active": True,
            "views": max(0, int(candidate.get("views") or 0)),
            "likes": max(0, int(candidate.get("likes") or 0)),
            "comments": max(0, int(candidate.get("comments") or 0)),
            "trend_score": _performance_signal(candidate),
            "duration_seconds": int(candidate.get("duration_seconds") or 0),
            "published_at": _parse_datetime(
                candidate.get("published_at"),
                field="published_at",
            ),
        }
        validate_training_eligibility_values(values)
        if dataset is None:
            dataset = DatasetContent(**values)
            db.add(dataset)
            stats["created"] += 1
        else:
            for field, value in values.items():
                setattr(dataset, field, value)
            stats["updated"] += 1
        db.flush()
        db.add(
            DatasetReviewEvent(
                collection_run_id=collection_run.collection_run_id,
                dataset_id=dataset.dataset_id,
                source_youtube_id=video_id,
                decision=decision,
                proposed_leaf_key=proposed_leaf,
                reviewed_leaf_key=reviewed_leaf,
                transcript_quality=quality,
                reviewer=reviewer,
                notes=notes,
                review_artifact_sha256=review_file_sha256,
                reviewed_at=reviewed_at,
            )
        )
        stats["approved"] += 1

    if errors:
        db.rollback()
        raise YouTubeCCDatasetError(
            "Review import failed; no rows were committed:\n- " + "\n- ".join(errors)
        )

    reviewed_video_ids = {
        str(video_id)
        for (video_id,) in (
            db.query(DatasetReviewEvent.source_youtube_id)
            .filter(DatasetReviewEvent.collection_run_id == collection_run.collection_run_id)
            .distinct()
            .all()
        )
    }
    reviewed_count = len(reviewed_video_ids & set(candidates))
    if reviewed_count == 0:
        collection_run.status = "review_pending"
    elif reviewed_count < len(candidates):
        collection_run.status = "partially_reviewed"
    else:
        collection_run.status = "reviewed"
    collection_run.candidate_artifact_path = str(candidate_file)
    collection_run.candidate_artifact_sha256 = candidate_file_sha256
    collection_run.completed_at = _utc_now().replace(tzinfo=None)
    db.commit()
    sync_taxonomy_registry(db)
    coverage = taxonomy_coverage(db)
    return {
        "status": "success",
        "run_key": run_key,
        "dataset_source": YOUTUBE_CC_DATASET_SOURCE,
        "dataset_version": dataset_version,
        "taxonomy_version": TAXONOMY_VERSION,
        "candidate_artifact_sha256": candidate_file_sha256,
        "review_artifact_sha256": review_file_sha256,
        "import_batch_id": import_batch_id,
        "reviewed_count": reviewed_count,
        "candidate_count": len(candidates),
        **dict(stats),
        "coverage": coverage,
    }


def _candidate_artifact_for_run(collection_run: DatasetCollectionRun) -> Path:
    raw_path = str(collection_run.candidate_artifact_path or "").strip()
    if not raw_path:
        raise YouTubeCCDatasetError(
            f"Collection run {collection_run.collection_run_id} has no candidate artifact"
        )
    path = Path(raw_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    if not path.is_file():
        raise YouTubeCCDatasetError(f"Candidate artifact not found: {path}")
    actual_hash = _sha256_file(path)
    if (
        collection_run.candidate_artifact_sha256
        and collection_run.candidate_artifact_sha256 != actual_hash
    ):
        raise YouTubeCCDatasetError(
            f"Candidate artifact hash mismatch for run {collection_run.collection_run_id}"
        )
    return path


def _candidate_review_status(
    event: DatasetReviewEvent | None,
) -> str:
    if event is None:
        return "pending"
    return "approved" if event.decision == "approve" else "rejected"


def _candidate_evidence_terms(candidate: dict[str, Any]) -> list[str]:
    leaf_key = normalize_taxonomy_leaf(candidate.get("proposed_leaf_key"))
    from app.services.taxonomy import taxonomy_profile_terms

    searchable = " ".join(
        (
            str(candidate.get("title") or ""),
            str(candidate.get("transcript") or ""),
        )
    ).lower()
    return [
        term
        for term in taxonomy_profile_terms(leaf_key)
        if len(term) > 1 and term.lower() in searchable
    ][:10]


def _serialize_review_candidate(
    *,
    collection_run: DatasetCollectionRun,
    candidate: dict[str, Any],
    event: DatasetReviewEvent | None,
    dataset: DatasetContent | None,
) -> dict[str, Any]:
    transcript = str(candidate.get("transcript") or "").strip()
    duration_seconds = int(candidate.get("duration_seconds") or 0)
    return {
        "collection_run_id": collection_run.collection_run_id,
        "dataset_version": collection_run.dataset_version,
        "source_youtube_id": str(candidate.get("source_youtube_id") or ""),
        "candidate_sha256": str(candidate.get("candidate_sha256") or ""),
        "title": str(candidate.get("title") or "Untitled"),
        "video_url": str(candidate.get("video_url") or ""),
        "channel_title": str(candidate.get("channel_title") or "Unknown channel"),
        "proposed_leaf_key": normalize_taxonomy_leaf(
            candidate.get("proposed_leaf_key")
        ),
        "transcript_language": str(candidate.get("transcript_language") or "und"),
        "caption_type": str(candidate.get("caption_type") or "unknown"),
        "duration_seconds": duration_seconds,
        "transcript": transcript,
        "transcript_preview": transcript[:700],
        "evidence_terms": _candidate_evidence_terms(candidate),
        "automated_checks": {
            "creative_commons": candidate.get("youtube_license_code") == "creativeCommon",
            "source_duration_present": duration_seconds > 0,
            "transcript_within_training_window": (
                0
                <= float(candidate.get("transcript_end_seconds") or 0.0)
                <= TRANSCRIPT_WINDOW_SECONDS
            ),
            "public_transcript": candidate.get("transcript_source")
            == YOUTUBE_CC_TRANSCRIPT_SOURCE,
            "supported_language": candidate.get("transcript_language")
            in SUPPORTED_TRANSCRIPT_LANGUAGES,
            "supported_caption": candidate.get("caption_type")
            in SUPPORTED_CAPTION_TYPES,
            "transcript_present": bool(transcript),
        },
        "dataset_usage": {
            "classification": True,
            "keyword_recommendation": True,
            "duration_recommendation": (
                0 < duration_seconds <= RECOMMENDATION_DURATION_MAX_SECONDS
            ),
        },
        "views": max(0, int(candidate.get("views") or 0)),
        "likes": max(0, int(candidate.get("likes") or 0)),
        "comments": max(0, int(candidate.get("comments") or 0)),
        "collection_strategy": str(
            candidate.get("collection_strategy")
            or CLASSIFICATION_DIVERSE_STRATEGY
        ),
        "average_views_per_day": float(
            candidate.get("average_views_per_day")
            or _performance_metrics(candidate)["average_views_per_day"]
        ),
        "engagement_rate": float(
            candidate.get("engagement_rate")
            or _performance_metrics(candidate)["engagement_rate"]
        ),
        "performance_rank_within_leaf": int(
            candidate.get("performance_rank_within_leaf") or 0
        ),
        "review_status": _candidate_review_status(event),
        "reviewed_leaf_key": event.reviewed_leaf_key if event else None,
        "transcript_quality": event.transcript_quality if event else None,
        "reviewer": event.reviewer if event else None,
        "reviewed_at": event.reviewed_at if event else None,
        "review_notes": event.notes if event else None,
        "dataset_id": dataset.dataset_id if dataset else None,
    }


def list_youtube_cc_review_queue(
    db: Session,
    *,
    limit: int = 20,
    offset: int = 0,
    review_status: str = "pending",
    leaf_key: str | None = None,
    collection_run_id: int | None = None,
    search: str | None = None,
) -> dict[str, Any]:
    query = (
        db.query(DatasetCollectionRun)
        .filter(DatasetCollectionRun.dataset_source == YOUTUBE_CC_DATASET_SOURCE)
        .order_by(DatasetCollectionRun.collection_run_id.desc())
    )
    if collection_run_id is not None:
        query = query.filter(
            DatasetCollectionRun.collection_run_id == collection_run_id
        )
    runs = query.all()
    run_ids = [run.collection_run_id for run in runs]

    latest_events: dict[tuple[int, str], DatasetReviewEvent] = {}
    if run_ids:
        events = (
            db.query(DatasetReviewEvent)
            .filter(DatasetReviewEvent.collection_run_id.in_(run_ids))
            .order_by(DatasetReviewEvent.review_event_id.asc())
            .all()
        )
        for event in events:
            latest_events[(event.collection_run_id, event.source_youtube_id)] = event

    datasets = {
        str(item.source_youtube_id): item
        for item in db.query(DatasetContent)
        .filter(DatasetContent.source_youtube_id.isnot(None))
        .all()
    }
    normalized_leaf = (
        normalize_taxonomy_leaf(leaf_key) if leaf_key and leaf_key != "all" else None
    )
    normalized_search = str(search or "").strip().lower()
    all_items: list[dict[str, Any]] = []
    run_items: list[dict[str, Any]] = []
    summary: Counter[str] = Counter()

    for run in runs:
        candidates = _load_candidates(
            _candidate_artifact_for_run(run),
            allow_empty=True,
        )
        try:
            run_config = json.loads(run.query_config_json)
        except (TypeError, json.JSONDecodeError):
            run_config = {}
        try:
            run_manifest = _load_collection_manifest(run, required=False)
        except YouTubeCCDatasetError:
            run_manifest = {}
        collection_progress = _collection_progress(
            candidates=list(candidates.values()),
            run_config=run_config,
        )
        run_summary: Counter[str] = Counter()
        for video_id, candidate in candidates.items():
            event = latest_events.get((run.collection_run_id, video_id))
            status = _candidate_review_status(event)
            run_summary[status] += 1
            proposed_leaf = normalize_taxonomy_leaf(candidate.get("proposed_leaf_key"))
            reviewed_leaf = normalize_taxonomy_leaf(
                event.reviewed_leaf_key if event else None
            )
            effective_leaf = (
                reviewed_leaf if reviewed_leaf in ACTIVE_LEAF_KEYS else proposed_leaf
            )
            if normalized_leaf and effective_leaf != normalized_leaf:
                continue
            if normalized_search:
                haystack = " ".join(
                    (
                        str(candidate.get("title") or ""),
                        str(candidate.get("channel_title") or ""),
                        str(candidate.get("transcript") or ""),
                        video_id,
                    )
                ).lower()
                if normalized_search not in haystack:
                    continue
            summary[status] += 1
            if review_status != "all" and status != review_status:
                continue
            all_items.append(
                _serialize_review_candidate(
                    collection_run=run,
                    candidate=candidate,
                    event=event,
                    dataset=datasets.get(video_id),
                )
            )
        run_items.append(
            {
                "collection_run_id": run.collection_run_id,
                "dataset_version": run.dataset_version,
                "status": run.status,
                "started_at": run.started_at,
                "total": len(candidates),
                "pending": int(run_summary["pending"]),
                "approved": int(run_summary["approved"]),
                "rejected": int(run_summary["rejected"]),
                "progress": collection_progress,
                "last_resumed_at": run.last_resumed_at,
                "failure_message": run_manifest.get("failure"),
            }
        )

    coverage = taxonomy_coverage(db)
    return {
        "total": len(all_items),
        "limit": limit,
        "offset": offset,
        "summary": {
            "total": int(sum(summary.values())),
            "pending": int(summary["pending"]),
            "approved": int(summary["approved"]),
            "rejected": int(summary["rejected"]),
        },
        "runs": run_items,
        "taxonomy": coverage["leaves"],
        "items": all_items[offset : offset + limit],
    }


def _single_review_csv(candidate: dict[str, Any], review: dict[str, Any]) -> str:
    row = {
        "candidate_sha256": candidate.get("candidate_sha256") or "",
        "source_youtube_id": candidate.get("source_youtube_id") or "",
        "title": candidate.get("title") or "",
        "video_url": candidate.get("video_url") or "",
        "channel_title": candidate.get("channel_title") or "",
        "proposed_leaf_key": candidate.get("proposed_leaf_key") or "",
        "transcript_language": candidate.get("transcript_language") or "",
        "caption_type": candidate.get("caption_type") or "",
        "duration_seconds": candidate.get("duration_seconds") or 0,
        "transcript_preview": str(candidate.get("transcript") or "")[:700],
        "decision": review["decision"],
        "reviewed_leaf_key": review.get("reviewed_leaf_key") or "",
        "transcript_quality": review.get("transcript_quality") or "",
        "reviewer": review["reviewer"],
        "reviewed_at": review["reviewed_at"],
        "review_notes": review.get("notes") or "",
    }
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=REVIEW_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerow(row)
    return output.getvalue()


def review_youtube_cc_candidate(
    db: Session,
    *,
    collection_run_id: int,
    source_youtube_id: str,
    decision: str,
    reviewer: str,
    reviewed_leaf_key: str | None = None,
    transcript_quality: str | None = None,
    notes: str | None = None,
    review_root: str | Path | None = None,
) -> dict[str, Any]:
    normalized_decision = str(decision or "").strip().lower()
    if normalized_decision not in {"approve", "reject"}:
        raise YouTubeCCDatasetError("decision must be approve or reject")
    reviewer = str(reviewer or "").strip()
    if not reviewer:
        raise YouTubeCCDatasetError("reviewer is required")

    collection_run = (
        db.query(DatasetCollectionRun)
        .filter(DatasetCollectionRun.collection_run_id == collection_run_id)
        .first()
    )
    if collection_run is None:
        raise YouTubeCCDatasetError(f"Collection run {collection_run_id} not found")
    candidate_path = _candidate_artifact_for_run(collection_run)
    candidates = _load_candidates(candidate_path)
    candidate = candidates.get(str(source_youtube_id))
    if candidate is None:
        raise YouTubeCCDatasetError(
            f"YouTube candidate {source_youtube_id} not found in run {collection_run_id}"
        )

    normalized_leaf = normalize_taxonomy_leaf(
        reviewed_leaf_key or candidate.get("proposed_leaf_key")
    )
    quality = str(transcript_quality or "").strip().lower()
    if normalized_decision == "approve":
        if normalized_leaf not in ACTIVE_LEAF_KEYS:
            raise YouTubeCCDatasetError("A valid reviewed_leaf_key is required")
        if quality not in ACCEPTED_TRANSCRIPT_QUALITIES:
            raise YouTubeCCDatasetError(
                "transcript_quality must be good or acceptable"
            )
    else:
        normalized_leaf = UNKNOWN_LEAF_KEY
        quality = ""

    latest_event = (
        db.query(DatasetReviewEvent)
        .filter(
            DatasetReviewEvent.collection_run_id == collection_run_id,
            DatasetReviewEvent.source_youtube_id == source_youtube_id,
        )
        .order_by(DatasetReviewEvent.review_event_id.desc())
        .first()
    )
    if (
        latest_event is not None
        and latest_event.decision == normalized_decision
        and normalize_taxonomy_leaf(latest_event.reviewed_leaf_key) == normalized_leaf
        and str(latest_event.transcript_quality or "") == quality
        and str(latest_event.notes or "") == str(notes or "")
    ):
        coverage = taxonomy_coverage(db)
        return {
            "status": "unchanged",
            "decision": normalized_decision,
            "source_youtube_id": source_youtube_id,
            "collection_run_id": collection_run_id,
            "dataset_id": latest_event.dataset_id,
            "review_event_id": latest_event.review_event_id,
            "run_status": collection_run.status,
            "coverage": coverage,
        }

    reviewed_at = _utc_now()
    event_root = (
        Path(review_root)
        if review_root is not None
        else Path(__file__).resolve().parents[2]
        / "data"
        / "reviews"
        / "youtube_cc"
        / collection_run.dataset_version
        / "events"
    )
    event_name = (
        f"review-{source_youtube_id}-{reviewed_at.strftime('%Y%m%dT%H%M%S%fZ')}.csv"
    )
    review_path = event_root / event_name
    _write_text_atomic(
        review_path,
        _single_review_csv(
            candidate,
            {
                "decision": normalized_decision,
                "reviewed_leaf_key": normalized_leaf,
                "transcript_quality": quality,
                "reviewer": reviewer,
                "reviewed_at": _iso_z(reviewed_at),
                "notes": str(notes or "").strip(),
            },
        ),
    )
    import_reviewed_youtube_cc_dataset(
        db,
        candidate_path=candidate_path,
        review_path=review_path,
    )
    latest_event = (
        db.query(DatasetReviewEvent)
        .filter(
            DatasetReviewEvent.collection_run_id == collection_run_id,
            DatasetReviewEvent.source_youtube_id == source_youtube_id,
        )
        .order_by(DatasetReviewEvent.review_event_id.desc())
        .first()
    )
    db.refresh(collection_run)
    return {
        "status": "success",
        "decision": normalized_decision,
        "source_youtube_id": source_youtube_id,
        "collection_run_id": collection_run_id,
        "dataset_id": latest_event.dataset_id if latest_event else None,
        "review_event_id": latest_event.review_event_id if latest_event else None,
        "run_status": collection_run.status,
        "coverage": taxonomy_coverage(db),
    }
