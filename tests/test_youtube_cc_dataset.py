import csv
import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

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
from app.services.taxonomy import TAXONOMY_VERSION, sync_taxonomy_registry
from app.services.youtube_cc_dataset import (
    REVIEW_FIELDS,
    YouTubeCCDatasetError,
    collect_youtube_cc_candidates,
    import_reviewed_youtube_cc_dataset,
    list_youtube_cc_review_queue,
    parse_iso8601_duration,
    review_youtube_cc_candidate,
)


def _search_response(video_ids):
    return {
        "items": [
            {"id": {"videoId": video_id}, "snippet": {"title": video_id}}
            for video_id in video_ids
        ]
    }


def _video(video_id, index=0):
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
        "contentDetails": {"duration": "PT2M10S", "caption": "true"},
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
        self.assertTrue(all(item["duration_seconds"] <= 300 for item in candidate_rows))
        self.assertTrue(all(not item["decision"] for item in review_rows))
        self.assertEqual(self.db.query(DatasetContent).count(), 0)
        run = self.db.query(DatasetCollectionRun).one()
        self.assertEqual(run.transcripts_collected, 3)

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
        with engine.connect() as connection:
            active = connection.execute(
                text("SELECT SUM(is_active), SUM(is_training_eligible) FROM dataset_contents")
            ).one()
        self.assertIn("transcript_source", columns)
        self.assertIn("reviewed_by", columns)
        self.assertEqual(result["legacy_rows_deactivated"], 2)
        self.assertEqual(tuple(active), (0, 0))
        engine.dispose()


if __name__ == "__main__":
    unittest.main()
