import hashlib
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database.db import Base
from app.database.models import (
    ClassificationModel,
    DatasetCollectionRun,
    DatasetContent,
    ModelEvaluationMetric,
)
from app.services.classification_training import (
    UNKNOWN_LEAF_KEY,
    classify_with_artifact,
    load_classification_artifact,
    predict_with_unknown,
    prepare_classification_dataset,
    train_and_evaluate_classification_models,
)
from app.services.dataset_contract import (
    SPLIT_STRATEGY,
    YOUTUBE_CC_LICENSE_NAME,
    YOUTUBE_CC_LICENSE_URL,
    channel_dataset_split,
)
from app.services.taxonomy import TAXONOMY_VERSION, taxonomy_path


HASH_A = hashlib.sha256(b"candidate artifact").hexdigest()
HASH_B = hashlib.sha256(b"review artifact").hexdigest()
HASH_C = hashlib.sha256(b"import batch").hexdigest()


class _ProbabilityEstimator:
    classes_ = np.asarray(["camera", "phone"], dtype=object)

    def predict_proba(self, texts):
        return np.asarray(
            [[0.55, 0.45] if "uncertain" in text else [0.05, 0.95] for text in texts]
        )


class ClassificationTrainingTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(self.engine)
        self.db = sessionmaker(bind=self.engine)()
        self.run = DatasetCollectionRun(
            run_key=hashlib.sha256(b"training-run").hexdigest(),
            dataset_source="youtube_cc",
            dataset_version="youtube-cc-training-test-v1",
            status="reviewed",
            region_code="TH",
            languages_json='["th", "en"]',
            query_config_json="{}",
        )
        self.db.add(self.run)
        self.db.commit()

    def tearDown(self):
        self.db.close()
        self.engine.dispose()

    @staticmethod
    def _channel_for_split(split, *, seed):
        index = 0
        while True:
            channel_id = f"channel-{seed}-{index}"
            actual, _group = channel_dataset_split(channel_id)
            if actual == split:
                return channel_id
            index += 1

    def _add_example(self, leaf_key, split, index, *, language="en"):
        path = taxonomy_path(leaf_key)
        channel_id = self._channel_for_split(split, seed=f"{leaf_key}-{split}-{index}")
        actual_split, creator_group_key = channel_dataset_split(channel_id)
        leaf_terms = {
            "phone": "smartphone mobile battery display camera android phone",
            "camera": "mirrorless camera lens photography sensor aperture camera",
        }
        transcript = " ".join([leaf_terms[leaf_key]] * 8) + f" sample {index}"
        youtube_id = f"{leaf_key}{split[:1]}{index:08d}"[:32]
        now = datetime.utcnow()
        self.db.add(
            DatasetContent(
                title=f"{leaf_key} review {index}",
                video_url=f"https://www.youtube.com/watch?v={youtube_id}",
                transcript=transcript,
                category=leaf_key,
                source_platform="youtube",
                dataset_source="youtube_cc",
                dataset_version="youtube-cc-training-test-v1",
                collection_run_id=self.run.collection_run_id,
                source_record_id=youtube_id,
                source_youtube_id=youtube_id,
                source_creator=f"Creator {channel_id}",
                source_channel_id=channel_id,
                source_category="28",
                source_subcategory=leaf_key,
                collection_query=f"{leaf_key} review",
                source_release_url=f"https://www.youtube.com/watch?v={youtube_id}",
                source_archive_sha256=HASH_A,
                source_annotation_path="data/reviews/training-test.csv",
                source_annotation_sha256=HASH_B,
                import_batch_id=HASH_C,
                taxonomy_version=TAXONOMY_VERSION,
                taxonomy_leaf_key=leaf_key,
                category_level_1=path["category_level_1"],
                category_level_2=path["category_level_2"],
                category_level_3=path["category_level_3"],
                language=language,
                verification_status="human_verified",
                label_source="human_review",
                license_name=YOUTUBE_CC_LICENSE_NAME,
                license_url=YOUTUBE_CC_LICENSE_URL,
                data_split=actual_split,
                split_strategy=SPLIT_STRATEGY,
                creator_group_key=creator_group_key,
                transcript_sha256=hashlib.sha256(transcript.encode()).hexdigest(),
                transcript_segment_count=8,
                transcript_start_seconds=0,
                transcript_end_seconds=180,
                transcript_window_seconds=300,
                transcript_source="youtube_public_caption",
                caption_type="manual",
                transcript_quality="good",
                reviewed_by="reviewer@example.test",
                reviewed_at=now,
                statistics_captured_at=now,
                license_verified_at=now,
                raw_metadata_json=json.dumps({"license": "creativeCommon"}),
                collection_strategy="classification_diverse",
                is_training_eligible=True,
                is_keyword_recommendation_eligible=True,
                is_duration_recommendation_eligible=True,
                is_active=True,
                duration_seconds=180,
                published_at=now,
            )
        )

    def _seed_ready_two_leaf_dataset(self):
        for leaf_key in ("phone", "camera"):
            self._add_example(leaf_key, "train", 0, language="th")
            self._add_example(leaf_key, "train", 1, language="en")
            self._add_example(leaf_key, "validation", 2, language="th")
            self._add_example(leaf_key, "test", 3, language="en")
        self.db.commit()

    def test_prepare_split_artifacts_group_channels_without_leakage(self):
        self._seed_ready_two_leaf_dataset()
        with tempfile.TemporaryDirectory() as temp_dir:
            prepared = prepare_classification_dataset(
                self.db,
                artifact_root=temp_dir,
                required_leaf_keys=("phone", "camera"),
                minimum_samples_per_leaf=4,
            )
            manifest_path = Path(prepared.report["artifacts"]["manifest_path"])
            split_paths = {
                split: Path(details["path"])
                for split, details in prepared.report["artifacts"]["splits"].items()
            }
            creator_splits = {}
            for split, path in split_paths.items():
                for line in path.read_text(encoding="utf-8").splitlines():
                    row = json.loads(line)
                    creator_splits.setdefault(row["creator_group_key"], set()).add(split)

            self.assertTrue(prepared.report["ready"])
            self.assertEqual(prepared.report["channel_leakage_count"], 0)
            self.assertTrue(manifest_path.is_file())
            self.assertTrue(all(len(splits) == 1 for splits in creator_splits.values()))
            self.assertEqual(
                prepared.report["unknown_support"]["strategy"],
                "confidence_rejection",
            )
            self.assertFalse(
                prepared.report["unknown_support"]["uses_synthetic_training_rows"]
            )

    def test_incomplete_dataset_stops_before_creating_database_models(self):
        self._add_example("phone", "train", 0)
        self.db.commit()
        with tempfile.TemporaryDirectory() as temp_dir:
            result = train_and_evaluate_classification_models(
                self.db,
                artifact_root=temp_dir,
                model_version="not-ready-v1",
                required_leaf_keys=("phone", "camera"),
                minimum_samples_per_leaf=4,
            )
        self.assertEqual(result["status"], "not_ready")
        self.assertFalse(result["dataset"]["ready"])
        self.assertEqual(self.db.query(ClassificationModel).count(), 0)
        self.assertEqual(self.db.query(ModelEvaluationMetric).count(), 0)

    def test_smoke_test_uses_incomplete_rows_and_is_never_promotion_eligible(self):
        self._seed_ready_two_leaf_dataset()
        with tempfile.TemporaryDirectory() as temp_dir:
            result = train_and_evaluate_classification_models(
                self.db,
                artifact_root=temp_dir,
                model_version="smoke-v1",
                required_leaf_keys=("phone", "camera"),
                minimum_samples_per_leaf=30,
                unknown_threshold=0.10,
                smoke_test=True,
            )
            models = self.db.query(ClassificationModel).all()
            best = next(
                item
                for item in result["models"]
                if item["model_id"] == result["best_model"]["model_id"]
            )
            prediction = classify_with_artifact(
                best["artifact_path"],
                title="Phone review",
                text="smartphone battery display camera android mobile",
            )

        self.assertEqual(result["status"], "smoke_test_evaluated")
        self.assertFalse(result["dataset"]["ready"])
        self.assertEqual(result["smoke_test_scope"]["dataset_sample_count"], 8)
        self.assertFalse(result["smoke_test_scope"]["promotion_eligible"])
        self.assertEqual(len(models), 3)
        self.assertTrue(all(model.status == "smoke_test_only" for model in models))
        self.assertTrue(all(not model.is_active for model in models))
        self.assertTrue(
            all(not item["qualification"]["passed"] for item in result["models"])
        )
        self.assertTrue(
            all(
                item["qualification"]["blocked_reason"]
                == "smoke_test_incomplete_dataset"
                for item in result["models"]
            )
        )
        self.assertTrue(prediction["smoke_test_only"])
        self.assertIn(prediction["taxonomy_leaf_key"], {"phone", "camera", "unknown"})
        reload_metrics = (
            self.db.query(ModelEvaluationMetric)
            .filter(ModelEvaluationMetric.metric_name == "reload_classify_passed")
            .all()
        )
        self.assertEqual(len(reload_metrics), 3)
        self.assertTrue(all(item.metric_value == 1.0 for item in reload_metrics))

    def test_benchmark_persists_all_metrics_but_never_activates_automatically(self):
        self._seed_ready_two_leaf_dataset()
        with tempfile.TemporaryDirectory() as temp_dir:
            result = train_and_evaluate_classification_models(
                self.db,
                artifact_root=temp_dir,
                model_version="benchmark-v1",
                required_leaf_keys=("phone", "camera"),
                minimum_samples_per_leaf=4,
                unknown_threshold=0.10,
                promotion_threshold=0.80,
            )
            models = self.db.query(ClassificationModel).all()
            metrics = self.db.query(ModelEvaluationMetric).all()
            artifact = load_classification_artifact(models[0].artifact_path)

        self.assertEqual(result["status"], "evaluated")
        self.assertEqual(result["database_models_created"], 3)
        self.assertEqual(len(models), 3)
        self.assertTrue(all(not model.is_active for model in models))
        self.assertTrue(any(model.status == "qualified" for model in models))
        self.assertTrue(
            all(
                model.status in {"qualified", "evaluated_below_threshold"}
                for model in models
            )
        )
        self.assertEqual(artifact["unknown_leaf_key"], UNKNOWN_LEAF_KEY)
        self.assertEqual(set(artifact["labels"]), {"phone", "camera"})
        self.assertGreater(len(metrics), 0)
        metric_names = {metric.metric_name for metric in metrics}
        self.assertTrue({"all", "th", "en"}.issubset({item.language for item in metrics}))
        self.assertTrue(
            {
                "accuracy",
                "precision_macro",
                "recall_macro",
                "f1_macro",
                "confusion_matrix",
                "passed",
            }.issubset(metric_names)
        )
        confusion = next(
            metric for metric in metrics if metric.metric_name == "confusion_matrix"
        )
        self.assertIn("unknown", json.loads(confusion.details)["labels"])

    def test_low_confidence_prediction_becomes_unknown(self):
        labels, confidences = predict_with_unknown(
            _ProbabilityEstimator(),
            ["uncertain topic", "clear phone"],
            unknown_threshold=0.60,
        )
        self.assertEqual(labels, [UNKNOWN_LEAF_KEY, "phone"])
        self.assertEqual(confidences, [0.55, 0.95])


if __name__ == "__main__":
    unittest.main()
