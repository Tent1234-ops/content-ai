import csv
import hashlib
import io
import json
import tempfile
import unittest
import urllib.error
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

from app.database.db import Base
from app.database.migrations import migrate_youtube_cc_dataset_schema
from app.database.models import (
    DatasetCollectionRun,
    DatasetContent,
    DatasetReviewEvent,
)
from app.services.dataset_eligibility import (
    production_transcript_query,
    validate_training_eligibility_values,
)
from app.services.taxonomy import (
    TAXONOMY_VERSION,
    collection_queries_for_leaf,
    sync_taxonomy_registry,
)
from app.services.youtube_cc_dataset import (
    CLASSIFICATION_DIVERSE_STRATEGY,
    RECOMMENDATION_HIGH_PERFORMANCE_STRATEGY,
    REVIEW_FIELDS,
    YouTubeCCDatasetError,
    YouTubeQuotaExceededError,
    YouTubeTranscriptProviderBlockedError,
    _youtube_get,
    collect_youtube_cc_candidates,
    import_reviewed_youtube_cc_dataset,
    list_youtube_cc_review_queue,
    parse_iso8601_duration,
    repair_quota_waiting_run_statuses,
    resume_youtube_cc_collection,
    review_youtube_cc_candidate,
)


def _search_response(video_ids):
    return {
        "items": [
            {"id": {"videoId": video_id}, "snippet": {"title": video_id}}
            for video_id in video_ids
        ]
    }


def _video(video_id, index=0, *, duration="PT2M10S"):
    return {
        "id": video_id,
        "snippet": {
            "title": f"รีวิวมือถือจริง {index}",
            "description": "รีวิวกล้อง แบตเตอรี่ และประสิทธิภาพ",
            "channelId": f"channel-{index % 2}",
            "channelTitle": f"Creator {index % 2}",
            "categoryId": "28",
            "publishedAt": "2025-01-01T00:00:00Z",
            "liveBroadcastContent": "none",
        },
        "contentDetails": {"duration": duration, "caption": "true"},
        "statistics": {
            "viewCount": str(10000 + index * 1000),
            "likeCount": str(500 + index * 10),
            "commentCount": str(40 + index),
        },
        "status": {"license": "creativeCommon"},
    }


def _transcript(video_id, _languages):
    transcript = f"วันนี้รีวิวมือถือ {video_id} กล้อง แบตเตอรี่ จอ และประสิทธิภาพ"
    return {
        "language": "th",
        "caption_type": "manual",
        "segments": [{"text": transcript, "start": 0.0, "duration": 30.0}],
        "segment_count": 1,
        "start_seconds": 0.0,
        "end_seconds": 30.0,
        "transcript": transcript,
        "transcript_sha256": hashlib.sha256(transcript.encode()).hexdigest(),
    }


class YouTubeCCDatasetTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(self.engine)
        self.db = sessionmaker(bind=self.engine)()
        sync_taxonomy_registry(self.db)

    def tearDown(self):
        self.db.close()
        self.engine.dispose()

    def _collect(self, root: Path, *, count: int = 3):
        ids = [f"realcc{i:05d}" for i in range(count)]

        def youtube_getter(resource, **_kwargs):
            if resource == "search":
                return _search_response(ids)
            if resource == "videos":
                return {"items": [_video(video_id, index) for index, video_id in enumerate(ids)]}
            raise AssertionError(resource)

        candidates = root / "candidates.jsonl"
        reviews = root / "review.csv"
        manifest = root / "manifest.json"
        result = collect_youtube_cc_candidates(
            self.db,
            api_key="test-key",
            candidate_path=candidates,
            review_path=reviews,
            manifest_path=manifest,
            dataset_version="youtube-cc-test-v1",
            leaf_keys=["phone"],
            target_per_leaf=count,
            languages=["th", "en"],
            max_pages_per_query=1,
            transcript_fetcher=_transcript,
            youtube_getter=youtube_getter,
        )
        return result, candidates, reviews, manifest

    @staticmethod
    def _fill_review(path: Path, decisions):
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row, decision in zip(rows, decisions):
            row["decision"] = decision
            if decision:
                row["reviewer"] = "human@example.test"
                row["reviewed_at"] = "2026-08-12T04:30:00Z"
                row["review_notes"] = "Transcript and category checked against the video."
            if decision == "approve":
                row["reviewed_leaf_key"] = "phone"
                row["transcript_quality"] = "good"
            elif decision == "reject":
                row["transcript_quality"] = "poor"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    def test_duration_parser_and_five_minute_boundary(self):
        self.assertEqual(parse_iso8601_duration("PT5M"), 300)
        self.assertEqual(parse_iso8601_duration("PT4M59S"), 299)
        self.assertEqual(parse_iso8601_duration("PT1H2M3S"), 3723)
        with self.assertRaises(YouTubeCCDatasetError):
            parse_iso8601_duration("not-a-duration")

    def test_phone_and_camera_queries_exclude_common_false_positives(self):
        phone_queries = collection_queries_for_leaf("phone")
        camera_queries = collection_queries_for_leaf("camera")
        self.assertTrue(any("-case" in query for query in phone_queries))
        self.assertTrue(any("-phone" in query for query in camera_queries))
        self.assertTrue(any("mirrorless" in query for query in camera_queries))

    def test_collection_creates_real_candidate_artifact_but_no_dataset_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result, candidates, reviews, manifest = self._collect(Path(temp_dir))
            candidate_rows = [
                json.loads(line)
                for line in candidates.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            with reviews.open("r", encoding="utf-8", newline="") as handle:
                review_rows = list(csv.DictReader(handle))

        self.assertEqual(result["status"], "collected")
        self.assertEqual(len(candidate_rows), 3)
        self.assertTrue(manifest.name.endswith(".json"))
        self.assertTrue(all(item["youtube_license_code"] == "creativeCommon" for item in candidate_rows))
        self.assertTrue(all(item["transcript_source"] == "youtube_public_caption" for item in candidate_rows))
        self.assertTrue(all(item["duration_seconds"] > 0 for item in candidate_rows))
        self.assertTrue(
            all(item["transcript_window_seconds"] == 300 for item in candidate_rows)
        )
        self.assertTrue(all(not item["decision"] for item in review_rows))
        self.assertEqual(self.db.query(DatasetContent).count(), 0)
        run = self.db.query(DatasetCollectionRun).one()
        self.assertEqual(run.transcripts_collected, 3)

    def test_long_source_video_is_trainable_but_not_duration_evidence(self):
        video_id = "longvideo01"

        def youtube_getter(resource, **kwargs):
            if resource == "search":
                return _search_response([video_id])
            if resource == "videos":
                return {"items": [_video(video_id, duration="PT1H2M3S")]}
            raise AssertionError(resource)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            candidates = root / "candidates.jsonl"
            reviews = root / "review.csv"
            collect_youtube_cc_candidates(
                self.db,
                api_key="test-key",
                candidate_path=candidates,
                review_path=reviews,
                manifest_path=root / "manifest.json",
                dataset_version="youtube-cc-long-source-v1",
                leaf_keys=["phone"],
                target_per_leaf=1,
                performance_target_per_leaf=0,
                languages=["th", "en"],
                max_pages_per_query=1,
                transcript_fetcher=_transcript,
                youtube_getter=youtube_getter,
            )
            candidate = json.loads(candidates.read_text(encoding="utf-8").strip())
            self.assertEqual(candidate["duration_seconds"], 3723)
            self.assertEqual(candidate["transcript_window_seconds"], 300)

            self._fill_review(reviews, ["approve"])
            import_reviewed_youtube_cc_dataset(
                self.db,
                candidate_path=candidates,
                review_path=reviews,
            )

        row = self.db.query(DatasetContent).one()
        self.assertEqual(row.duration_seconds, 3723)
        self.assertTrue(row.is_training_eligible)
        self.assertTrue(row.is_keyword_recommendation_eligible)
        self.assertFalse(row.is_duration_recommendation_eligible)
        self.assertEqual(production_transcript_query(self.db).count(), 1)

    def test_collection_uses_separate_diverse_and_high_performance_samples(self):
        ids = [f"strategy{i:03d}" for i in range(4)]
        search_orders = []

        def youtube_getter(resource, **kwargs):
            if resource == "search":
                search_orders.append(kwargs["order"])
                return _search_response(ids)
            if resource == "videos":
                requested = kwargs["id"].split(",")
                return {
                    "items": [
                        _video(video_id, ids.index(video_id))
                        for video_id in requested
                    ]
                }
            raise AssertionError(resource)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            result = collect_youtube_cc_candidates(
                self.db,
                api_key="test-key",
                candidate_path=root / "candidates.jsonl",
                review_path=root / "review.csv",
                manifest_path=root / "manifest.json",
                dataset_version="youtube-cc-strategy-v1",
                leaf_keys=["phone"],
                target_per_leaf=4,
                performance_target_per_leaf=2,
                languages=["th", "en"],
                max_pages_per_query=1,
                transcript_fetcher=_transcript,
                youtube_getter=youtube_getter,
            )
            rows = [
                json.loads(line)
                for line in (root / "candidates.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]

        self.assertEqual(result["status"], "collected")
        self.assertEqual(search_orders[:2], ["viewCount", "relevance"])
        counts = Counter(row["collection_strategy"] for row in rows)
        self.assertEqual(counts[RECOMMENDATION_HIGH_PERFORMANCE_STRATEGY], 2)
        self.assertEqual(counts[CLASSIFICATION_DIVERSE_STRATEGY], 2)
        self.assertTrue(all(row["average_views_per_day"] > 0 for row in rows))
        self.assertEqual(
            sorted(row["performance_rank_within_leaf"] for row in rows),
            [1, 2, 3, 4],
        )

    def test_collection_reserves_minimum_thai_transcripts_per_leaf(self):
        ids = ["en00000001", "en00000002", "en00000003", "en00000004", "th00000001", "th00000002"]

        def youtube_getter(resource, **kwargs):
            if resource == "search":
                return _search_response(ids)
            if resource == "videos":
                items = []
                for index, video_id in enumerate(kwargs["id"].split(",")):
                    item = _video(video_id, index)
                    item["snippet"]["channelId"] = f"language-channel-{index}"
                    items.append(item)
                return {"items": items}
            raise AssertionError(resource)

        def transcript_fetcher(video_id, _languages):
            language = "th" if video_id.startswith("th") else "en"
            transcript = f"{language} transcript {video_id}"
            return {
                "language": language,
                "caption_type": "manual",
                "segments": [{"text": transcript, "start": 0.0, "duration": 20.0}],
                "segment_count": 1,
                "start_seconds": 0.0,
                "end_seconds": 20.0,
                "transcript": transcript,
                "transcript_sha256": hashlib.sha256(transcript.encode()).hexdigest(),
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            result = collect_youtube_cc_candidates(
                self.db,
                api_key="test-key",
                candidate_path=root / "candidates.jsonl",
                review_path=root / "review.csv",
                manifest_path=root / "manifest.json",
                dataset_version="youtube-cc-language-balance-v1",
                leaf_keys=["phone"],
                target_per_leaf=4,
                performance_target_per_leaf=0,
                languages=["th", "en"],
                min_thai_per_leaf=2,
                max_videos_per_channel_per_leaf=3,
                max_pages_per_query=1,
                transcript_fetcher=transcript_fetcher,
                youtube_getter=youtube_getter,
            )
            rows = [
                json.loads(line)
                for line in (root / "candidates.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]

        languages = Counter(row["transcript_language"] for row in rows)
        self.assertEqual(result["status"], "collected")
        self.assertEqual(languages, Counter({"en": 2, "th": 2}))
        self.assertEqual(
            result["quality_filters"]["skipped_by_reason"][
                "non_thai_capacity_reserved"
            ],
            2,
        )
        leaf_progress = result["progress"]["by_leaf"][0]
        self.assertEqual(leaf_progress["thai_minimum"], 2)
        self.assertEqual(leaf_progress["language_counts"]["th"], 2)

    def test_collection_caps_each_channel_within_a_leaf(self):
        ids = [f"channelcap{i:02d}" for i in range(5)]

        def youtube_getter(resource, **kwargs):
            if resource == "search":
                return _search_response(ids)
            if resource == "videos":
                items = []
                for index, video_id in enumerate(kwargs["id"].split(",")):
                    item = _video(video_id, index)
                    item["snippet"]["channelId"] = (
                        "repeated-channel" if index < 3 else f"unique-channel-{index}"
                    )
                    items.append(item)
                return {"items": items}
            raise AssertionError(resource)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            result = collect_youtube_cc_candidates(
                self.db,
                api_key="test-key",
                candidate_path=root / "candidates.jsonl",
                review_path=root / "review.csv",
                manifest_path=root / "manifest.json",
                dataset_version="youtube-cc-channel-cap-v1",
                leaf_keys=["phone"],
                target_per_leaf=4,
                performance_target_per_leaf=0,
                languages=["th", "en"],
                min_thai_per_leaf=0,
                max_videos_per_channel_per_leaf=2,
                max_pages_per_query=1,
                transcript_fetcher=_transcript,
                youtube_getter=youtube_getter,
            )
            rows = [
                json.loads(line)
                for line in (root / "candidates.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]

        channels = Counter(row["channel_id"] for row in rows)
        self.assertEqual(result["status"], "collected")
        self.assertLessEqual(max(channels.values()), 2)
        self.assertEqual(len(channels), 3)
        self.assertEqual(
            result["quality_filters"]["skipped_by_reason"][
                "channel_cap_reached"
            ],
            1,
        )

    def test_quota_exhaustion_checkpoints_as_quota_waiting(self):
        callbacks = []

        def youtube_getter(resource, **_kwargs):
            raise YouTubeQuotaExceededError(resource, 429, "Quota exceeded")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaises(YouTubeQuotaExceededError):
                collect_youtube_cc_candidates(
                    self.db,
                    api_key="test-key",
                    candidate_path=root / "candidates.jsonl",
                    review_path=root / "review.csv",
                    manifest_path=root / "manifest.json",
                    dataset_version="youtube-cc-quota-v1",
                    leaf_keys=["phone"],
                    target_per_leaf=1,
                    performance_target_per_leaf=0,
                    languages=["th", "en"],
                    min_thai_per_leaf=0,
                    max_pages_per_query=1,
                    transcript_fetcher=_transcript,
                    youtube_getter=youtube_getter,
                    progress_callback=lambda manifest: callbacks.append(
                        manifest["status"]
                    ),
                )
            manifest = json.loads(
                (root / "manifest.json").read_text(encoding="utf-8")
            )
            run = self.db.query(DatasetCollectionRun).one()
            self.assertEqual(run.status, "quota_waiting")
            self.assertIsNone(run.completed_at)
            self.assertEqual(run.errors_count, 0)
            self.assertEqual(callbacks, ["running", "quota_waiting"])
            self.assertTrue(manifest["quota"]["retryable"])

            run.status = "failed"
            run.errors_count = 1
            run.completed_at = datetime.now()
            self.db.commit()
            repaired = repair_quota_waiting_run_statuses(self.db)
            self.db.refresh(run)

        self.assertEqual(repaired["run_ids"], [run.collection_run_id])
        self.assertEqual(run.status, "quota_waiting")
        self.assertEqual(run.errors_count, 0)
        self.assertIsNone(run.completed_at)

    def test_http_429_is_recognized_as_quota_exhaustion(self):
        error = urllib.error.HTTPError(
            "https://example.test",
            429,
            "Too Many Requests",
            {},
            io.BytesIO(b'{"error":{"message":"Quota exceeded"}}'),
        )
        with patch("urllib.request.urlopen", side_effect=error):
            with self.assertRaises(YouTubeQuotaExceededError):
                _youtube_get("search", api_key="test", timeout_seconds=1, q="phone")

    def test_transcript_ip_block_checkpoints_without_advancing_search_page(self):
        video_id = "blocked001"
        callbacks = []

        def youtube_getter(resource, **_kwargs):
            if resource == "search":
                return {
                    **_search_response([video_id]),
                    "nextPageToken": "NEXT_PAGE",
                }
            if resource == "videos":
                return {"items": [_video(video_id)]}
            raise AssertionError(resource)

        def blocked_transcript(requested_video_id, _languages):
            raise YouTubeTranscriptProviderBlockedError(
                requested_video_id,
                "IP blocked",
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaises(YouTubeTranscriptProviderBlockedError):
                collect_youtube_cc_candidates(
                    self.db,
                    api_key="test-key",
                    candidate_path=root / "candidates.jsonl",
                    review_path=root / "review.csv",
                    manifest_path=root / "manifest.json",
                    dataset_version="youtube-cc-transcript-wait-v1",
                    leaf_keys=["phone"],
                    target_per_leaf=1,
                    performance_target_per_leaf=0,
                    languages=["th", "en"],
                    min_thai_per_leaf=0,
                    max_pages_per_query=1,
                    transcript_fetcher=blocked_transcript,
                    youtube_getter=youtube_getter,
                    progress_callback=lambda manifest: callbacks.append(
                        manifest["status"]
                    ),
                )

            manifest = json.loads(
                (root / "manifest.json").read_text(encoding="utf-8")
            )
            run = self.db.query(DatasetCollectionRun).one()

        self.assertEqual(run.status, "transcript_waiting")
        self.assertIsNone(run.completed_at)
        self.assertEqual(run.transcripts_collected, 0)
        self.assertEqual(manifest["search_state"], {})
        self.assertTrue(manifest["transcript_provider"]["retryable"])
        self.assertEqual(callbacks, ["running", "transcript_waiting"])

    def test_empty_schema_three_run_upgrades_quality_controls_on_resume(self):
        video_id = "upgraded001"

        def quota_getter(resource, **_kwargs):
            raise YouTubeQuotaExceededError(resource, 429, "Quota exceeded")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaises(YouTubeQuotaExceededError):
                collect_youtube_cc_candidates(
                    self.db,
                    api_key="test-key",
                    candidate_path=root / "candidates.jsonl",
                    review_path=root / "review.csv",
                    manifest_path=root / "manifest.json",
                    dataset_version="youtube-cc-upgrade-v1",
                    leaf_keys=["phone"],
                    target_per_leaf=1,
                    performance_target_per_leaf=0,
                    languages=["th", "en"],
                    min_thai_per_leaf=0,
                    max_pages_per_query=1,
                    transcript_fetcher=_transcript,
                    youtube_getter=quota_getter,
                )
            run = self.db.query(DatasetCollectionRun).one()
            config = json.loads(run.query_config_json)
            for field in (
                "min_thai_per_leaf",
                "max_videos_per_channel_per_leaf",
                "language_balance_policy",
                "channel_diversity_policy",
                "queries_by_leaf",
            ):
                config.pop(field, None)
            config["schema_version"] = 3
            run.run_key = "legacy-empty-run-key"
            run.query_config_json = json.dumps(config)
            self.db.commit()

            def youtube_getter(resource, **kwargs):
                if resource == "search":
                    return _search_response([video_id])
                if resource == "videos":
                    return {"items": [_video(video_id)]}
                raise AssertionError(resource)

            resumed = resume_youtube_cc_collection(
                self.db,
                collection_run_id=run.collection_run_id,
                api_key="test-key",
                max_pages_per_query=1,
                transcript_fetcher=_transcript,
                youtube_getter=youtube_getter,
            )

        self.db.refresh(run)
        upgraded_config = json.loads(run.query_config_json)
        self.assertEqual(resumed["status"], "collected")
        self.assertEqual(upgraded_config["schema_version"], 4)
        self.assertEqual(upgraded_config["min_thai_per_leaf"], 1)
        self.assertEqual(upgraded_config["max_videos_per_channel_per_leaf"], 3)
        self.assertIn("-case", " ".join(upgraded_config["queries_by_leaf"]["phone"]))

    def test_collection_deduplicates_candidates_across_runs(self):
        responses = [
            ["existing001"],
            ["existing001", "fresh00001"],
        ]
        collection_number = 0

        def make_getter(response_ids):
            def youtube_getter(resource, **kwargs):
                if resource == "search":
                    return _search_response(response_ids)
                if resource == "videos":
                    requested = kwargs["id"].split(",")
                    return {
                        "items": [
                            _video(video_id, response_ids.index(video_id))
                            for video_id in requested
                        ]
                    }
                raise AssertionError(resource)

            return youtube_getter

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for response_ids in responses:
                collection_number += 1
                collect_youtube_cc_candidates(
                    self.db,
                    api_key="test-key",
                    candidate_path=root / f"candidates-{collection_number}.jsonl",
                    review_path=root / f"review-{collection_number}.csv",
                    manifest_path=root / f"manifest-{collection_number}.json",
                    dataset_version=f"youtube-cc-dedup-v{collection_number}",
                    leaf_keys=["phone"],
                    target_per_leaf=1,
                    performance_target_per_leaf=0,
                    languages=["th", "en"],
                    max_pages_per_query=1,
                    transcript_fetcher=_transcript,
                    youtube_getter=make_getter(response_ids),
                )
            second_manifest = json.loads(
                (root / "manifest-2.json").read_text(encoding="utf-8")
            )
            second_rows = [
                json.loads(line)
                for line in (root / "candidates-2.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]

        self.assertEqual(
            [row["source_youtube_id"] for row in second_rows],
            ["fresh00001"],
        )
        self.assertGreaterEqual(
            second_manifest["deduplication"]["skipped_by_reason"][
                "duplicate_previous_video"
            ],
            1,
        )
        self.assertEqual(self.db.query(DatasetCollectionRun).count(), 2)

    def test_channel_cap_counts_candidates_from_prior_runs(self):
        first_ids = ["priorchannel1", "priorchannel2"]
        second_ids = ["priorchannel3", "otherchannel1"]

        def make_getter(response_ids):
            def youtube_getter(resource, **kwargs):
                if resource == "search":
                    return _search_response(response_ids)
                if resource == "videos":
                    items = []
                    for index, video_id in enumerate(kwargs["id"].split(",")):
                        item = _video(video_id, index)
                        item["snippet"]["channelId"] = (
                            "shared-channel"
                            if video_id != "otherchannel1"
                            else "other-channel"
                        )
                        items.append(item)
                    return {"items": items}
                raise AssertionError(resource)

            return youtube_getter

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            collect_youtube_cc_candidates(
                self.db,
                api_key="test-key",
                candidate_path=root / "candidates-1.jsonl",
                review_path=root / "review-1.csv",
                manifest_path=root / "manifest-1.json",
                dataset_version="youtube-cc-channel-history-v1",
                leaf_keys=["phone"],
                target_per_leaf=2,
                performance_target_per_leaf=0,
                languages=["th", "en"],
                min_thai_per_leaf=0,
                max_videos_per_channel_per_leaf=2,
                max_pages_per_query=1,
                transcript_fetcher=_transcript,
                youtube_getter=make_getter(first_ids),
            )
            second = collect_youtube_cc_candidates(
                self.db,
                api_key="test-key",
                candidate_path=root / "candidates-2.jsonl",
                review_path=root / "review-2.csv",
                manifest_path=root / "manifest-2.json",
                dataset_version="youtube-cc-channel-history-v2",
                leaf_keys=["phone"],
                target_per_leaf=1,
                performance_target_per_leaf=0,
                languages=["th", "en"],
                min_thai_per_leaf=0,
                max_videos_per_channel_per_leaf=2,
                max_pages_per_query=1,
                transcript_fetcher=_transcript,
                youtube_getter=make_getter(second_ids),
            )
            rows = [
                json.loads(line)
                for line in (root / "candidates-2.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]

        self.assertEqual([row["source_youtube_id"] for row in rows], ["otherchannel1"])
        self.assertGreaterEqual(
            second["quality_filters"]["skipped_by_reason"][
                "channel_cap_reached"
            ],
            1,
        )

    def test_partial_collection_resumes_from_saved_page_token(self):
        first_id = "resume00001"
        second_id = "resume00002"
        page_tokens = []

        def youtube_getter(resource, **kwargs):
            if resource == "search":
                page_token = kwargs.get("pageToken")
                page_tokens.append(page_token)
                if page_token == "next-page":
                    return _search_response([second_id])
                return {
                    **_search_response([first_id]),
                    "nextPageToken": "next-page",
                }
            if resource == "videos":
                requested = kwargs["id"].split(",")
                return {
                    "items": [
                        _video(video_id, 0 if video_id == first_id else 1)
                        for video_id in requested
                    ]
                }
            raise AssertionError(resource)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = collect_youtube_cc_candidates(
                self.db,
                api_key="test-key",
                candidate_path=root / "candidates.jsonl",
                review_path=root / "review.csv",
                manifest_path=root / "manifest.json",
                dataset_version="youtube-cc-resume-v1",
                leaf_keys=["phone"],
                target_per_leaf=2,
                performance_target_per_leaf=0,
                languages=["th", "en"],
                max_pages_per_query=1,
                transcript_fetcher=_transcript,
                youtube_getter=youtube_getter,
            )
            run = self.db.query(DatasetCollectionRun).one()
            self.assertEqual(first["status"], "partial")
            resumed = resume_youtube_cc_collection(
                self.db,
                collection_run_id=run.collection_run_id,
                api_key="test-key",
                max_pages_per_query=1,
                transcript_fetcher=_transcript,
                youtube_getter=youtube_getter,
            )
            rows = [
                json.loads(line)
                for line in (root / "candidates.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]

        self.assertEqual(resumed["status"], "collected")
        self.assertEqual({row["source_youtube_id"] for row in rows}, {first_id, second_id})
        self.assertIn("next-page", page_tokens)
        self.db.refresh(run)
        self.assertEqual(run.resume_count, 1)
        self.assertEqual(run.transcripts_collected, 2)

    def test_resume_refuses_to_mutate_a_human_reviewed_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _result, _candidates, _reviews, _manifest = self._collect(root, count=1)
            run = self.db.query(DatasetCollectionRun).one()
            pending = list_youtube_cc_review_queue(self.db)["items"][0]
            review_youtube_cc_candidate(
                self.db,
                collection_run_id=run.collection_run_id,
                source_youtube_id=pending["source_youtube_id"],
                decision="approve",
                reviewer="admin@example.test [user:1]",
                reviewed_leaf_key="phone",
                transcript_quality="good",
                review_root=root / "events",
            )
            with self.assertRaisesRegex(YouTubeCCDatasetError, "immutable"):
                resume_youtube_cc_collection(
                    self.db,
                    collection_run_id=run.collection_run_id,
                    api_key="test-key",
                    transcript_fetcher=_transcript,
                    youtube_getter=lambda *_args, **_kwargs: {},
                )

    def test_resume_refuses_legacy_five_minute_source_filter(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self._collect(Path(temp_dir), count=1)
            run = self.db.query(DatasetCollectionRun).one()
            config = json.loads(run.query_config_json)
            config.pop("source_video_duration_policy")
            run.query_config_json = json.dumps(config)
            original_status = run.status
            self.db.commit()

            with self.assertRaisesRegex(YouTubeCCDatasetError, "legacy five-minute"):
                resume_youtube_cc_collection(
                    self.db,
                    collection_run_id=run.collection_run_id,
                    api_key="test-key",
                    transcript_fetcher=_transcript,
                    youtube_getter=lambda *_args, **_kwargs: self.fail(
                        "legacy resume must not call YouTube"
                    ),
                )

            self.db.refresh(run)
            self.assertEqual(run.status, original_status)
            self.assertEqual(run.resume_count, 0)

    def test_import_requires_explicit_human_review_and_audits_rejection(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            _result, candidates, reviews, _manifest = self._collect(Path(temp_dir))
            pending = import_reviewed_youtube_cc_dataset(
                self.db,
                candidate_path=candidates,
                review_path=reviews,
            )
            self.assertEqual(pending["pending"], 3)
            self.assertEqual(self.db.query(DatasetContent).count(), 0)
            self.assertEqual(self.db.query(DatasetCollectionRun).one().status, "review_pending")

            self._fill_review(reviews, ["approve", "reject", ""])
            imported = import_reviewed_youtube_cc_dataset(
                self.db,
                candidate_path=candidates,
                review_path=reviews,
            )

        self.assertEqual(imported["approved"], 1)
        self.assertEqual(imported["rejected"], 1)
        self.assertEqual(imported["pending"], 1)
        self.assertEqual(self.db.query(DatasetCollectionRun).one().status, "partially_reviewed")
        self.assertEqual(self.db.query(DatasetContent).count(), 1)
        self.assertEqual(self.db.query(DatasetReviewEvent).count(), 2)
        row = self.db.query(DatasetContent).one()
        self.assertEqual(row.dataset_source, "youtube_cc")
        self.assertEqual(row.taxonomy_version, TAXONOMY_VERSION)
        self.assertEqual(row.verification_status, "human_verified")
        self.assertEqual(row.label_source, "human_review")
        self.assertTrue(row.is_training_eligible)
        self.assertTrue(row.is_keyword_recommendation_eligible)
        self.assertTrue(row.is_duration_recommendation_eligible)
        self.assertEqual(production_transcript_query(self.db).count(), 1)

    def test_import_is_atomic_when_any_review_is_invalid(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            _result, candidates, reviews, _manifest = self._collect(Path(temp_dir), count=2)
            self._fill_review(reviews, ["approve", "approve"])
            with reviews.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            rows[1]["reviewer"] = ""
            with reviews.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(YouTubeCCDatasetError, "no rows were committed"):
                import_reviewed_youtube_cc_dataset(
                    self.db,
                    candidate_path=candidates,
                    review_path=reviews,
                )
        self.assertEqual(self.db.query(DatasetContent).count(), 0)
        self.assertEqual(self.db.query(DatasetReviewEvent).count(), 0)

    def test_admin_review_queue_approves_and_rejects_without_editing_csv(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            _result, _candidates, _reviews, _manifest = self._collect(root, count=2)
            run = self.db.query(DatasetCollectionRun).one()

            pending = list_youtube_cc_review_queue(self.db)
            self.assertEqual(pending["summary"]["pending"], 2)
            self.assertEqual(pending["items"][0]["review_status"], "pending")

            first = pending["items"][0]
            approved = review_youtube_cc_candidate(
                self.db,
                collection_run_id=run.collection_run_id,
                source_youtube_id=first["source_youtube_id"],
                decision="approve",
                reviewer="admin@example.test [user:1]",
                reviewed_leaf_key="phone",
                transcript_quality="good",
                notes="Video and transcript checked.",
                review_root=root / "events",
            )
            self.assertEqual(approved["decision"], "approve")
            self.assertEqual(approved["run_status"], "partially_reviewed")
            self.assertIsNotNone(approved["dataset_id"])

            remaining = list_youtube_cc_review_queue(self.db)
            self.assertEqual(remaining["summary"]["pending"], 1)
            second = remaining["items"][0]
            rejected = review_youtube_cc_candidate(
                self.db,
                collection_run_id=run.collection_run_id,
                source_youtube_id=second["source_youtube_id"],
                decision="reject",
                reviewer="admin@example.test [user:1]",
                notes="Transcript does not match the target category.",
                review_root=root / "events",
            )

            self.assertEqual(rejected["decision"], "reject")
            self.assertEqual(rejected["run_status"], "reviewed")
            self.assertEqual(self.db.query(DatasetContent).count(), 1)
            self.assertEqual(self.db.query(DatasetReviewEvent).count(), 2)
            approved_queue = list_youtube_cc_review_queue(
                self.db, review_status="approved"
            )
            self.assertEqual(approved_queue["total"], 1)
            self.assertEqual(approved_queue["items"][0]["reviewed_leaf_key"], "phone")

    def test_eligibility_rejects_source_verified_or_unreviewed_rows(self):
        values = {
            "is_training_eligible": True,
            "dataset_source": "youtube_cc",
            "verification_status": "source_verified",
            "label_source": "search_query",
        }
        with self.assertRaisesRegex(ValueError, "human_verified"):
            validate_training_eligibility_values(values)

    def test_migration_adds_youtube_cc_columns_and_disables_old_sources(self):
        engine = create_engine("sqlite:///:memory:")
        with engine.begin() as connection:
            connection.execute(
                text(
                    "CREATE TABLE dataset_collection_runs ("
                    "collection_run_id INTEGER PRIMARY KEY)"
                )
            )
            connection.execute(
                text(
                    "CREATE TABLE dataset_contents ("
                    "dataset_id INTEGER PRIMARY KEY, source_platform VARCHAR(50), "
                    "dataset_source VARCHAR(100), is_active BOOLEAN, "
                    "is_training_eligible BOOLEAN)"
                )
            )
            connection.execute(
                text(
                    "INSERT INTO dataset_contents VALUES "
                    "(1, 'youtube', 'mmsum', 1, 1), "
                    "(2, 'youtube_seed', 'demo_seed', 1, 1)"
                )
            )
        result = migrate_youtube_cc_dataset_schema(engine)
        columns = {item["name"] for item in inspect(engine).get_columns("dataset_contents")}
        run_columns = {
            item["name"]
            for item in inspect(engine).get_columns("dataset_collection_runs")
        }
        with engine.connect() as connection:
            active = connection.execute(
                text("SELECT SUM(is_active), SUM(is_training_eligible) FROM dataset_contents")
            ).one()
        self.assertIn("transcript_source", columns)
        self.assertIn("reviewed_by", columns)
        self.assertIn("collection_strategy", columns)
        self.assertIn("average_views_per_day", columns)
        self.assertIn("review_artifact_path", run_columns)
        self.assertIn("resume_count", run_columns)
        self.assertEqual(result["legacy_rows_deactivated"], 2)
        self.assertEqual(tuple(active), (0, 0))
        engine.dispose()


if __name__ == "__main__":
    unittest.main()
