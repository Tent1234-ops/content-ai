import os
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database.db import Base
from app.database.models import DatasetContent
from app.routes.analyze import _build_recommendation
from app.services.classification import classify_text_domain
from app.services.pipeline.core import normalize_asr_terms
from app.services.pipeline.domain_rules import normalize_domain
from app.services.recommendation import build_recommendation_from_analysis_data
from models.speech_to_text import check_model_readiness, transcribe_with_meta
from scripts.seed_demo_dataset import build_rows


class Phase12AiEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(self.engine)
        self.db = sessionmaker(bind=self.engine)()

    def tearDown(self):
        self.db.close()
        self.engine.dispose()

    def _seed_rows(self):
        self.db.add_all(DatasetContent(**row) for row in build_rows())
        self.db.commit()

    def test_seed_has_24_rows_for_every_required_category(self):
        counts = Counter(str(row["category"]) for row in build_rows())
        self.assertEqual(
            counts,
            {
                "smartphone": 24,
                "food_drink": 24,
                "skincare": 24,
                "audio": 24,
                "keyboard": 24,
                "mouse": 24,
                "fashion": 24,
            },
        )

    def test_category_aliases_share_one_taxonomy(self):
        self.assertEqual(normalize_domain("food"), "food_drink")
        self.assertEqual(normalize_domain("Food & Drink"), "food_drink")
        self.assertEqual(normalize_domain("mechanical keyboard"), "keyboard")
        self.assertEqual(normalize_domain("skin care"), "skincare")

    def test_keyboard_recommendation_uses_exact_rows_and_complete_evidence(self):
        self._seed_rows()
        result = build_recommendation_from_analysis_data(
            self.db,
            domain="keyboard",
            user_keywords=["typing feel", "build quality"],
            dimension_status=[],
            hook_terms=[],
            source_prefix="youtube",
            profile_limit=80,
        )

        profile = result["dataset_profile"]
        evidence = result["evidence"]
        self.assertEqual(result["domain"], "keyboard")
        self.assertEqual(profile["sample_size"], 24)
        self.assertEqual(
            evidence["dataset_sample_size"],
            sum(evidence["source_platform_counts"].values()),
        )
        self.assertGreater(len(evidence["exemplar_titles"]), 0)
        self.assertEqual(evidence["duration_sample_size"], 24)
        self.assertEqual(len(evidence["duration_samples"]), 24)
        self.assertEqual(len(evidence["dataset_row_ids"]), 24)
        self.assertNotIn("fallback rules", evidence["data_source_label"].lower())

    def test_classification_confidence_is_not_forced_to_72_percent(self):
        profiles = [
            {
                "domain": "keyboard",
                "sample_size": 5,
                "term_weights": {"keyboard": 1.0},
            },
            {
                "domain": "audio",
                "sample_size": 5,
                "term_weights": {"sound": 1.0},
            },
        ]
        with (
            patch(
                "app.services.classification.build_domain_classifier_profiles",
                return_value=profiles,
            ),
            patch("app.services.classification._cosine", side_effect=[0.4, 0.35]),
        ):
            result = classify_text_domain(
                MagicMock(),
                text="mechanical keyboard sound",
                title="keyboard review",
            )

        self.assertEqual(result["domain"], "keyboard")
        self.assertLess(result["confidence"], 0.72)

    def test_stt_auto_detect_does_not_force_thai(self):
        segment = SimpleNamespace(text="keyboard review", no_speech_prob=0.1, start=0, end=1)
        info = SimpleNamespace(language="en", language_probability=0.94)
        model = MagicMock()
        model.transcribe.return_value = ([segment], info)
        with (
            patch("models.speech_to_text.ModelManager.get_model", return_value=model),
            patch.dict(os.environ, {"ASR_LANGUAGE": "auto"}),
        ):
            result = transcribe_with_meta("sample.wav", model_size="small")

        kwargs = model.transcribe.call_args.kwargs
        self.assertNotIn("language", kwargs)
        self.assertEqual(result["language"], "en")
        self.assertEqual(result["language_probability"], 0.94)

    def test_thai_spell_correction_is_disabled_by_default(self):
        with (
            patch(
                "app.services.pipeline.core.pythai_correct",
                side_effect=AssertionError("unbounded spell correction must not run"),
            ),
            patch.dict(
                os.environ,
                {"ANALYZE_ENABLE_THAI_SPELL_CORRECTION": "0"},
            ),
        ):
            result = normalize_asr_terms("ทดสอบเสียงจากหูฟัง", aggressive=True)

        self.assertTrue(result)

    def test_model_readiness_checks_required_local_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = os.path.join(temp_dir, "small")
            os.makedirs(model_dir)
            with patch.dict(os.environ, {"ASR_MODEL_DIR": temp_dir}):
                missing = check_model_readiness("small")
                for name in ("config.json", "model.bin", "tokenizer.json"):
                    Path(model_dir, name).touch()
                ready = check_model_readiness("small")

        self.assertFalse(missing["ready"])
        self.assertTrue(ready["ready"])

    def test_filename_fallback_caps_confidence_and_returns_warning(self):
        self._seed_rows()
        result = {
            "transcript": "keyboard review",
            "analysis": {
                "domain": "keyboard",
                "top_keywords": [{"keyword": "typing feel", "score": 1.0}],
                "dimension_status": [],
                "stt_meta": {
                    "transcript_source": "fallback_filename",
                    "fallback_reason": "stt_failed",
                    "warning": "Filename fallback warning",
                },
            },
        }
        recommendation, _ = _build_recommendation(
            self.db,
            filename="keyboard_review.mp4",
            result=result,
        )

        self.assertLessEqual(recommendation["classification"]["confidence"], 0.25)
        self.assertEqual(recommendation["classification"]["input_source"], "filename_fallback")
        self.assertEqual(recommendation["evidence"]["warning"], "Filename fallback warning")


if __name__ == "__main__":
    unittest.main()
