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
from collections import Counter
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
    MAX_VIDEO_DURATION_SECONDS,
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
    except ImportError as exc:
        raise YouTubeCCDatasetError(
            "youtube-transcript-api is not installed; run pip install -r requirements.txt"
        ) from exc

    try:
        tracks = list(YouTubeTranscriptApi().list(video_id))
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
) -> dict[str, Any]:
    snippet = item.get("snippet") or {}
    content_details = item.get("contentDetails") or {}
    statistics = item.get("statistics") or {}
    status = item.get("status") or {}
    video_id = str(item.get("id") or "").strip()
    duration_seconds = parse_iso8601_duration(str(content_details.get("duration") or ""))
    candidate = {
        "schema_version": 1,
        "run_key": run_key,
        "dataset_source": YOUTUBE_CC_DATASET_SOURCE,
        "dataset_version": dataset_version,
        "taxonomy_version": TAXONOMY_VERSION,
        "proposed_leaf_key": leaf_key,
        "collection_query": query,
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
    languages: Sequence[str] = SUPPORTED_TRANSCRIPT_LANGUAGES,
    region_code: str = "TH",
    max_pages_per_query: int = 2,
    timeout_seconds: float = 10.0,
    transcript_fetcher: Callable[[str, Sequence[str]], dict[str, Any]] = fetch_public_transcript,
    youtube_getter: Callable[..., dict[str, Any]] = _youtube_get,
) -> dict[str, Any]:
    normalized_leaves = tuple(dict.fromkeys(normalize_taxonomy_leaf(item) for item in leaf_keys))
    if not normalized_leaves or any(item not in ACTIVE_LEAF_KEYS for item in normalized_leaves):
        raise YouTubeCCDatasetError("All requested leaves must be active taxonomy leaves")
    normalized_languages = tuple(dict.fromkeys(str(item).lower() for item in languages))
    if not normalized_languages or any(
        item not in SUPPORTED_TRANSCRIPT_LANGUAGES for item in normalized_languages
    ):
        raise YouTubeCCDatasetError(
            f"Languages must be selected from {SUPPORTED_TRANSCRIPT_LANGUAGES}"
        )
    if target_per_leaf < 1:
        raise YouTubeCCDatasetError("target_per_leaf must be at least 1")

    started_at = _utc_now()
    run_config = {
        "dataset_version": dataset_version,
        "leaf_keys": normalized_leaves,
        "target_per_leaf": target_per_leaf,
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
        candidate_artifact_path=str(candidate_path),
        manifest_path=str(manifest_path),
        started_at=started_at.replace(tzinfo=None),
    )
    existing = (
        db.query(DatasetCollectionRun)
        .filter(DatasetCollectionRun.run_key == run_key)
        .first()
    )
    if existing is not None:
        raise YouTubeCCDatasetError(f"Collection run already exists: {run_key}")
    db.add(collection_run)
    db.commit()

    candidates: list[dict[str, Any]] = []
    seen_video_ids: set[str] = set()
    seen_transcript_hashes: set[str] = set()
    accepted_by_leaf: Counter[str] = Counter()
    rejected_reasons: Counter[str] = Counter()
    candidates_seen = 0

    try:
        for leaf_key in normalized_leaves:
            for query in collection_queries_for_leaf(leaf_key):
                if accepted_by_leaf[leaf_key] >= target_per_leaf:
                    break
                page_token: str | None = None
                query_language = "th" if re.search(r"[\u0E00-\u0E7F]", query) else "en"
                for _page in range(max_pages_per_query):
                    search_params: dict[str, Any] = {
                        "part": "snippet",
                        "q": query,
                        "type": "video",
                        "videoLicense": "creativeCommon",
                        "videoCaption": "closedCaption",
                        "maxResults": 50,
                        "order": "relevance",
                        "regionCode": region_code.upper(),
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
                    video_ids = [
                        str((item.get("id") or {}).get("videoId") or "").strip()
                        for item in search.get("items") or []
                    ]
                    video_ids = [item for item in video_ids if item and item not in seen_video_ids]
                    if video_ids:
                        details = youtube_getter(
                            "videos",
                            api_key=api_key,
                            timeout_seconds=timeout_seconds,
                            part="snippet,contentDetails,statistics,status",
                            id=",".join(video_ids),
                            maxResults=50,
                        )
                        for item in details.get("items") or []:
                            candidates_seen += 1
                            video_id = str(item.get("id") or "").strip()
                            seen_video_ids.add(video_id)
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
                            if not (0 < duration <= MAX_VIDEO_DURATION_SECONDS):
                                rejected_reasons["outside_5_minute_scope"] += 1
                                continue
                            if str(content_details.get("caption") or "").lower() != "true":
                                rejected_reasons["no_caption_flag"] += 1
                                continue
                            if str(snippet.get("liveBroadcastContent") or "none") != "none":
                                rejected_reasons["live_broadcast"] += 1
                                continue
                            try:
                                transcript = transcript_fetcher(video_id, normalized_languages)
                            except Exception:
                                rejected_reasons["transcript_unavailable"] += 1
                                continue
                            transcript_hash = str(transcript.get("transcript_sha256") or "")
                            if transcript_hash in seen_transcript_hashes:
                                rejected_reasons["duplicate_transcript"] += 1
                                continue
                            candidate = _make_candidate(
                                item=item,
                                transcript=transcript,
                                run_key=run_key,
                                dataset_version=dataset_version,
                                leaf_key=leaf_key,
                                query=query,
                                region_code=region_code.upper(),
                                collected_at=started_at,
                            )
                            candidates.append(candidate)
                            seen_transcript_hashes.add(transcript_hash)
                            accepted_by_leaf[leaf_key] += 1
                            if accepted_by_leaf[leaf_key] >= target_per_leaf:
                                break
                    if accepted_by_leaf[leaf_key] >= target_per_leaf:
                        break
                    page_token = str(search.get("nextPageToken") or "").strip() or None
                    if not page_token:
                        break

        candidate_file = Path(candidate_path)
        review_file = Path(review_path)
        manifest_file = Path(manifest_path)
        jsonl = "".join(_canonical_json(item) + "\n" for item in candidates)
        _write_text_atomic(candidate_file, jsonl)
        _write_text_atomic(review_file, _review_csv_text(candidates))
        candidate_file_sha256 = _sha256_file(candidate_file)
        review_file_sha256 = _sha256_file(review_file)
        completed_at = _utc_now()
        status = (
            "collected"
            if all(accepted_by_leaf[item] >= target_per_leaf for item in normalized_leaves)
            else "partial"
        )
        manifest = {
            "schema_version": 1,
            "run_key": run_key,
            "dataset_source": YOUTUBE_CC_DATASET_SOURCE,
            "dataset_version": dataset_version,
            "taxonomy_version": TAXONOMY_VERSION,
            "status": status,
            "started_at": _iso_z(started_at),
            "completed_at": _iso_z(completed_at),
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
            "rejected_reasons": dict(rejected_reasons),
        }
        _write_text_atomic(
            manifest_file,
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        )

        collection_run.status = status
        collection_run.candidate_artifact_sha256 = candidate_file_sha256
        collection_run.manifest_sha256 = _sha256_file(manifest_file)
        collection_run.candidates_seen = candidates_seen
        collection_run.transcripts_collected = len(candidates)
        collection_run.errors_count = sum(rejected_reasons.values())
        collection_run.completed_at = completed_at.replace(tzinfo=None)
        db.commit()
        return manifest
    except Exception:
        db.rollback()
        failed_run = (
            db.query(DatasetCollectionRun)
            .filter(DatasetCollectionRun.run_key == run_key)
            .first()
        )
        if failed_run is not None:
            failed_run.status = "failed"
            failed_run.candidates_seen = candidates_seen
            failed_run.transcripts_collected = len(candidates)
            failed_run.errors_count = sum(rejected_reasons.values()) + 1
            failed_run.completed_at = _utc_now().replace(tzinfo=None)
            db.commit()
        raise


def _load_candidates(path: Path) -> dict[str, dict[str, Any]]:
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
    if not candidates:
        raise YouTubeCCDatasetError(f"Candidate artifact is empty: {path}")
    return candidates


def _split_for_channel(channel_id: str) -> tuple[str, str]:
    creator_group_key = _sha256_text(channel_id)
    bucket = int(creator_group_key[:8], 16) % 100
    if bucket < 70:
        split = "train"
    elif bucket < 85:
        split = "validation"
    else:
        split = "test"
    return split, creator_group_key


def _performance_signal(candidate: dict[str, Any]) -> float:
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
    return round(
        math.log1p(average_views_per_day) * (1.0 + min(engagement_rate * 10.0, 1.0)),
        6,
    )


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
                candidate.get("raw_metadata") or {},
                ensure_ascii=False,
                sort_keys=True,
            ),
            "is_training_eligible": True,
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
            "within_five_minutes": 0 < duration_seconds <= MAX_VIDEO_DURATION_SECONDS,
            "public_transcript": candidate.get("transcript_source")
            == YOUTUBE_CC_TRANSCRIPT_SOURCE,
            "supported_language": candidate.get("transcript_language")
            in SUPPORTED_TRANSCRIPT_LANGUAGES,
            "supported_caption": candidate.get("caption_type")
            in SUPPORTED_CAPTION_TYPES,
            "transcript_present": bool(transcript),
        },
        "views": max(0, int(candidate.get("views") or 0)),
        "likes": max(0, int(candidate.get("likes") or 0)),
        "comments": max(0, int(candidate.get("comments") or 0)),
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
        candidates = _load_candidates(_candidate_artifact_for_run(run))
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
