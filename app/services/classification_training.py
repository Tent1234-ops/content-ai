from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sqlalchemy.orm import Session

from app.database.models import (
    ClassificationModel,
    DatasetContent,
    ModelEvaluationMetric,
    SystemLog,
)
from app.services.dataset_contract import (
    PRODUCTION_SPLITS,
    SPLIT_STRATEGY,
    YOUTUBE_CC_DATASET_SOURCE,
    channel_dataset_split,
)
from app.services.dataset_eligibility import production_transcript_query
from app.services.taxonomy import (
    ACTIVE_LEAF_KEYS,
    MIN_VERIFIED_SAMPLES,
    TAXONOMY_VERSION,
    UNKNOWN_LEAF_KEY,
)


MODEL_FAMILY = "taxonomy-text-classifier"
MODEL_PROMOTION_THRESHOLD = 0.80
UNKNOWN_CONFIDENCE_THRESHOLD = 0.60
RANDOM_STATE = 42
OVERALL_LEAF_KEY = "__overall__"
OVERALL_LANGUAGE = "all"


class ClassificationTrainingError(RuntimeError):
    pass


@dataclass(frozen=True)
class TrainingExample:
    dataset_id: int
    source_record_id: str
    source_youtube_id: str
    source_channel_id: str
    creator_group_key: str
    dataset_version: str
    split: str
    language: str
    leaf_key: str
    title: str
    transcript: str
    transcript_sha256: str

    @property
    def model_text(self) -> str:
        # Repeating the title gives concise, human-reviewed topic evidence more weight.
        return f"{self.title}\n{self.title}\n{self.transcript}".strip()

    def artifact_record(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "source_record_id": self.source_record_id,
            "source_youtube_id": self.source_youtube_id,
            "source_channel_id": self.source_channel_id,
            "creator_group_key": self.creator_group_key,
            "dataset_version": self.dataset_version,
            "data_split": self.split,
            "language": self.language,
            "taxonomy_leaf_key": self.leaf_key,
            "title": self.title,
            "transcript": self.transcript,
            "transcript_sha256": self.transcript_sha256,
        }


@dataclass(frozen=True)
class PreparedClassificationDataset:
    examples: tuple[TrainingExample, ...]
    report: dict[str, Any]
    artifact_dir: Path | None


@dataclass(frozen=True)
class ClassificationModelSpec:
    model_key: str
    model_type: str
    description: str
    factory: Callable[[], Pipeline]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_version(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-.")
    if not normalized:
        raise ClassificationTrainingError("model_version cannot be empty")
    return normalized[:100]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Sequence[TrainingExample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row.artifact_record(), ensure_ascii=False, sort_keys=True)
                + "\n"
            )
    temporary.replace(path)


def _write_joblib(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(payload, temporary)
    temporary.replace(path)


def _dataset_fingerprint(examples: Sequence[TrainingExample]) -> str:
    canonical_rows = [
        {
            "dataset_id": item.dataset_id,
            "source_record_id": item.source_record_id,
            "creator_group_key": item.creator_group_key,
            "dataset_version": item.dataset_version,
            "split": item.split,
            "language": item.language,
            "leaf_key": item.leaf_key,
            "transcript_sha256": item.transcript_sha256,
        }
        for item in sorted(examples, key=lambda value: value.dataset_id)
    ]
    return _sha256_text(
        json.dumps(canonical_rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


def _load_training_examples(
    db: Session,
    *,
    required_leaf_keys: Sequence[str],
) -> list[TrainingExample]:
    rows = (
        production_transcript_query(db)
        .filter(DatasetContent.taxonomy_leaf_key.in_(tuple(required_leaf_keys)))
        .order_by(DatasetContent.dataset_id.asc())
        .all()
    )
    examples: list[TrainingExample] = []
    for row in rows:
        examples.append(
            TrainingExample(
                dataset_id=int(row.dataset_id),
                source_record_id=str(row.source_record_id or ""),
                source_youtube_id=str(row.source_youtube_id or ""),
                source_channel_id=str(row.source_channel_id or ""),
                creator_group_key=str(row.creator_group_key or ""),
                dataset_version=str(row.dataset_version or ""),
                split=str(row.data_split or ""),
                language=str(row.language or "und"),
                leaf_key=str(row.taxonomy_leaf_key or ""),
                title=str(row.title or ""),
                transcript=str(row.transcript or ""),
                transcript_sha256=str(row.transcript_sha256 or ""),
            )
        )
    return examples


def _dataset_readiness(
    examples: Sequence[TrainingExample],
    *,
    required_leaf_keys: Sequence[str],
    minimum_samples_per_leaf: int,
) -> dict[str, Any]:
    split_counts: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()
    channels_by_split: dict[str, set[str]] = defaultdict(set)
    channel_splits: dict[str, set[str]] = defaultdict(set)
    leaf_split_counts: dict[str, Counter[str]] = defaultdict(Counter)
    leaf_language_counts: dict[str, Counter[str]] = defaultdict(Counter)
    leaf_channels: dict[str, set[str]] = defaultdict(set)
    issues: list[str] = []

    for item in examples:
        split_counts[item.split] += 1
        language_counts[item.language] += 1
        channels_by_split[item.split].add(item.creator_group_key)
        channel_splits[item.creator_group_key].add(item.split)
        leaf_split_counts[item.leaf_key][item.split] += 1
        leaf_language_counts[item.leaf_key][item.language] += 1
        leaf_channels[item.leaf_key].add(item.creator_group_key)

        try:
            expected_split, expected_group = channel_dataset_split(
                item.source_channel_id
            )
        except ValueError as exc:
            issues.append(f"dataset_id={item.dataset_id}: {exc}")
            continue
        if item.split != expected_split:
            issues.append(
                f"dataset_id={item.dataset_id}: split={item.split} does not match "
                f"{SPLIT_STRATEGY} ({expected_split})"
            )
        if item.creator_group_key != expected_group:
            issues.append(
                f"dataset_id={item.dataset_id}: creator_group_key does not match channel"
            )

    leaked_channels = sorted(
        channel for channel, splits in channel_splits.items() if len(splits) > 1
    )
    if leaked_channels:
        issues.append(
            f"{len(leaked_channels)} channel group(s) leak across dataset splits"
        )

    by_leaf: list[dict[str, Any]] = []
    minimum_train_samples = max(2, math.ceil(minimum_samples_per_leaf * 0.50))
    minimum_evaluation_samples = max(
        1, math.ceil(minimum_samples_per_leaf * 0.10)
    )
    for leaf_key in required_leaf_keys:
        counts = leaf_split_counts[leaf_key]
        total = sum(counts.values())
        leaf_issues: list[str] = []
        if total < minimum_samples_per_leaf:
            leaf_issues.append(
                f"requires {minimum_samples_per_leaf - total} more verified sample(s)"
            )
        if counts["train"] < minimum_train_samples:
            leaf_issues.append(
                f"requires at least {minimum_train_samples} training samples"
            )
        if counts["validation"] < minimum_evaluation_samples:
            leaf_issues.append(
                "requires at least "
                f"{minimum_evaluation_samples} validation samples"
            )
        if counts["test"] < minimum_evaluation_samples:
            leaf_issues.append(
                f"requires at least {minimum_evaluation_samples} test samples"
            )
        if leaf_issues:
            issues.append(f"{leaf_key}: " + "; ".join(leaf_issues))
        by_leaf.append(
            {
                "leaf_key": leaf_key,
                "total": total,
                "minimum_required": minimum_samples_per_leaf,
                "minimum_split_counts": {
                    "train": minimum_train_samples,
                    "validation": minimum_evaluation_samples,
                    "test": minimum_evaluation_samples,
                },
                "split_counts": {
                    split: int(counts[split]) for split in PRODUCTION_SPLITS
                },
                "language_counts": dict(sorted(leaf_language_counts[leaf_key].items())),
                "unique_channels": len(leaf_channels[leaf_key]),
                "ready": not leaf_issues,
                "issues": leaf_issues,
            }
        )

    return {
        "ready": not issues,
        "issues": list(dict.fromkeys(issues)),
        "sample_count": len(examples),
        "split_counts": {
            split: int(split_counts[split]) for split in PRODUCTION_SPLITS
        },
        "language_counts": dict(sorted(language_counts.items())),
        "channel_counts": {
            split: len(channels_by_split[split]) for split in PRODUCTION_SPLITS
        },
        "unique_channels": len(channel_splits),
        "channel_leakage_count": len(leaked_channels),
        "leaked_creator_group_keys": leaked_channels,
        "by_leaf": by_leaf,
    }


def prepare_classification_dataset(
    db: Session,
    *,
    artifact_root: str | Path | None = None,
    required_leaf_keys: Sequence[str] | None = None,
    minimum_samples_per_leaf: int = MIN_VERIFIED_SAMPLES,
) -> PreparedClassificationDataset:
    leaves = tuple(dict.fromkeys(required_leaf_keys or ACTIVE_LEAF_KEYS))
    invalid_leaves = sorted(set(leaves) - set(ACTIVE_LEAF_KEYS))
    if invalid_leaves:
        raise ClassificationTrainingError(
            "Unknown or non-trainable taxonomy leaves: " + ", ".join(invalid_leaves)
        )
    if minimum_samples_per_leaf < 1:
        raise ClassificationTrainingError("minimum_samples_per_leaf must be positive")

    examples = _load_training_examples(db, required_leaf_keys=leaves)
    fingerprint = _dataset_fingerprint(examples)
    readiness = _dataset_readiness(
        examples,
        required_leaf_keys=leaves,
        minimum_samples_per_leaf=minimum_samples_per_leaf,
    )
    dataset_versions = sorted({item.dataset_version for item in examples})
    report: dict[str, Any] = {
        "status": "ready" if readiness["ready"] else "not_ready",
        "ready": bool(readiness["ready"]),
        "dataset_source": YOUTUBE_CC_DATASET_SOURCE,
        "dataset_versions": dataset_versions,
        "dataset_fingerprint": fingerprint,
        "taxonomy_version": TAXONOMY_VERSION,
        "required_leaf_keys": list(leaves),
        "minimum_samples_per_leaf": minimum_samples_per_leaf,
        "split_strategy": SPLIT_STRATEGY,
        "unknown_support": {
            "leaf_key": UNKNOWN_LEAF_KEY,
            "strategy": "confidence_rejection",
            "uses_synthetic_training_rows": False,
        },
        **readiness,
        "artifacts": {},
    }

    artifact_dir: Path | None = None
    if artifact_root is not None:
        artifact_dir = Path(artifact_root) / f"dataset-{fingerprint[:16]}"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        split_artifacts: dict[str, dict[str, Any]] = {}
        for split in PRODUCTION_SPLITS:
            path = artifact_dir / f"{split}.jsonl"
            split_rows = [item for item in examples if item.split == split]
            _write_jsonl(path, split_rows)
            split_artifacts[split] = {
                "path": str(path.resolve()),
                "sha256": _sha256_file(path),
                "sample_count": len(split_rows),
            }
        report["artifacts"] = {
            "directory": str(artifact_dir.resolve()),
            "splits": split_artifacts,
        }
        manifest_path = artifact_dir / "dataset_manifest.json"
        report["artifacts"]["manifest_path"] = str(manifest_path.resolve())
        _write_json(manifest_path, report)
        report["artifacts"]["manifest_sha256"] = _sha256_file(manifest_path)

    return PreparedClassificationDataset(
        examples=tuple(examples),
        report=report,
        artifact_dir=artifact_dir,
    )


def _word_char_features() -> FeatureUnion:
    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=40_000,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=1,
                    max_features=40_000,
                    sublinear_tf=True,
                ),
            ),
        ]
    )


def default_classification_model_specs() -> tuple[ClassificationModelSpec, ...]:
    return (
        ClassificationModelSpec(
            model_key="taxonomy-tfidf-logreg",
            model_type="tfidf_word_char_logistic_regression",
            description="Word and character TF-IDF with balanced logistic regression",
            factory=lambda: Pipeline(
                [
                    ("features", _word_char_features()),
                    (
                        "classifier",
                        LogisticRegression(
                            max_iter=2_000,
                            class_weight="balanced",
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
        ),
        ClassificationModelSpec(
            model_key="taxonomy-tfidf-complement-nb",
            model_type="tfidf_word_char_complement_nb",
            description="Word and character TF-IDF with Complement Naive Bayes",
            factory=lambda: Pipeline(
                [
                    ("features", _word_char_features()),
                    ("classifier", ComplementNB(alpha=0.5)),
                ]
            ),
        ),
        ClassificationModelSpec(
            model_key="taxonomy-tfidf-sgd-log",
            model_type="tfidf_word_char_sgd_logistic",
            description="Word and character TF-IDF with balanced SGD logistic loss",
            factory=lambda: Pipeline(
                [
                    ("features", _word_char_features()),
                    (
                        "classifier",
                        SGDClassifier(
                            loss="log_loss",
                            class_weight="balanced",
                            max_iter=3_000,
                            tol=1e-4,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            ),
        ),
    )


def predict_with_unknown(
    estimator: Pipeline,
    texts: Sequence[str],
    *,
    unknown_threshold: float = UNKNOWN_CONFIDENCE_THRESHOLD,
) -> tuple[list[str], list[float]]:
    if not 0.0 < unknown_threshold < 1.0:
        raise ClassificationTrainingError("unknown_threshold must be between 0 and 1")
    probabilities = np.asarray(estimator.predict_proba(list(texts)), dtype=float)
    classes = np.asarray(estimator.classes_, dtype=object)
    if probabilities.ndim != 2 or probabilities.shape[1] != len(classes):
        raise ClassificationTrainingError("classifier returned invalid probabilities")
    top_indexes = probabilities.argmax(axis=1)
    confidences = probabilities[np.arange(len(top_indexes)), top_indexes]
    predictions = [
        str(classes[index]) if confidence >= unknown_threshold else UNKNOWN_LEAF_KEY
        for index, confidence in zip(top_indexes, confidences)
    ]
    return predictions, [round(float(value), 6) for value in confidences]


def _evaluate_examples(
    estimator: Pipeline,
    examples: Sequence[TrainingExample],
    *,
    labels: Sequence[str],
    unknown_threshold: float,
) -> dict[str, Any]:
    y_true = [item.leaf_key for item in examples]
    y_pred, confidences = predict_with_unknown(
        estimator,
        [item.model_text for item in examples],
        unknown_threshold=unknown_threshold,
    )
    all_labels = list(labels)
    true_label_set = set(y_true)
    supported_labels = [label for label in all_labels if label in true_label_set]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=all_labels,
        average=None,
        zero_division=0,
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=supported_labels,
        average="macro",
        zero_division=0,
    )
    matrix_labels = all_labels + [UNKNOWN_LEAF_KEY]
    matrix = confusion_matrix(y_true, y_pred, labels=matrix_labels)
    per_class = {
        label: {
            "precision": round(float(precision[index]), 6),
            "recall": round(float(recall[index]), 6),
            "f1": round(float(f1[index]), 6),
            "support": int(support[index]),
        }
        for index, label in enumerate(all_labels)
    }
    unknown_count = sum(1 for value in y_pred if value == UNKNOWN_LEAF_KEY)
    return {
        "sample_size": len(examples),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "precision_macro": round(float(precision_macro), 6),
        "recall_macro": round(float(recall_macro), 6),
        "f1_macro": round(float(f1_macro), 6),
        "mean_confidence": round(float(np.mean(confidences)), 6),
        "unknown_prediction_count": unknown_count,
        "unknown_rate": round(unknown_count / max(1, len(examples)), 6),
        "labels": matrix_labels,
        "metric_labels": supported_labels,
        "confusion_matrix": matrix.astype(int).tolist(),
        "per_class": per_class,
    }


def _evaluation_by_language(
    estimator: Pipeline,
    examples: Sequence[TrainingExample],
    *,
    labels: Sequence[str],
    unknown_threshold: float,
) -> dict[str, dict[str, Any]]:
    result = {
        OVERALL_LANGUAGE: _evaluate_examples(
            estimator,
            examples,
            labels=labels,
            unknown_threshold=unknown_threshold,
        )
    }
    for language in sorted({item.language for item in examples}):
        language_rows = [item for item in examples if item.language == language]
        result[language] = _evaluate_examples(
            estimator,
            language_rows,
            labels=labels,
            unknown_threshold=unknown_threshold,
        )
    return result


def _metric_rows(
    *,
    model_id: int,
    dataset_split: str,
    evaluations: dict[str, dict[str, Any]],
) -> list[ModelEvaluationMetric]:
    rows: list[ModelEvaluationMetric] = []
    scalar_names = (
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "mean_confidence",
        "unknown_rate",
    )
    for language, result in evaluations.items():
        sample_size = int(result["sample_size"])
        for metric_name in scalar_names:
            rows.append(
                ModelEvaluationMetric(
                    model_id=model_id,
                    dataset_split=dataset_split,
                    language=language,
                    taxonomy_level=3,
                    taxonomy_leaf_key=OVERALL_LEAF_KEY,
                    metric_name=metric_name,
                    metric_value=float(result[metric_name]),
                    sample_size=sample_size,
                    details=None,
                )
            )
        rows.append(
            ModelEvaluationMetric(
                model_id=model_id,
                dataset_split=dataset_split,
                language=language,
                taxonomy_level=3,
                taxonomy_leaf_key=OVERALL_LEAF_KEY,
                metric_name="confusion_matrix",
                metric_value=float(result["accuracy"]),
                sample_size=sample_size,
                details=json.dumps(
                    {
                        "labels": result["labels"],
                        "matrix": result["confusion_matrix"],
                        "unknown_prediction_count": result[
                            "unknown_prediction_count"
                        ],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            )
        )

    overall = evaluations[OVERALL_LANGUAGE]
    for leaf_key, metrics in overall["per_class"].items():
        for metric_name in ("precision", "recall", "f1"):
            rows.append(
                ModelEvaluationMetric(
                    model_id=model_id,
                    dataset_split=dataset_split,
                    language=OVERALL_LANGUAGE,
                    taxonomy_level=3,
                    taxonomy_leaf_key=leaf_key,
                    metric_name=metric_name,
                    metric_value=float(metrics[metric_name]),
                    sample_size=int(metrics["support"]),
                    details=None,
                )
            )
    return rows


def _dataset_version_for_model(report: dict[str, Any]) -> str:
    versions = list(report.get("dataset_versions") or [])
    if len(versions) == 1:
        return str(versions[0])[:100]
    fingerprint = str(report["dataset_fingerprint"])
    return f"mixed-{fingerprint[:16]}"


def _qualification(
    validation: dict[str, Any],
    test: dict[str, Any],
    *,
    promotion_threshold: float,
) -> dict[str, Any]:
    checks = {
        "validation_accuracy": float(validation["accuracy"]),
        "validation_macro_f1": float(validation["f1_macro"]),
        "test_accuracy": float(test["accuracy"]),
        "test_macro_f1": float(test["f1_macro"]),
    }
    passed = all(value >= promotion_threshold for value in checks.values())
    return {
        "passed": passed,
        "threshold": promotion_threshold,
        "checks": checks,
        "policy": (
            "validation_accuracy, validation_macro_f1, test_accuracy and "
            "test_macro_f1 must all meet the threshold"
        ),
        "activation": "manual_after_runtime_integration" if passed else "blocked",
    }


def train_and_evaluate_classification_models(
    db: Session,
    *,
    artifact_root: str | Path,
    model_version: str | None = None,
    required_leaf_keys: Sequence[str] | None = None,
    minimum_samples_per_leaf: int = MIN_VERIFIED_SAMPLES,
    unknown_threshold: float = UNKNOWN_CONFIDENCE_THRESHOLD,
    promotion_threshold: float = MODEL_PROMOTION_THRESHOLD,
    model_specs: Sequence[ClassificationModelSpec] | None = None,
    prepare_only: bool = False,
    smoke_test: bool = False,
) -> dict[str, Any]:
    if not 0.0 < promotion_threshold <= 1.0:
        raise ClassificationTrainingError(
            "promotion_threshold must be greater than 0 and at most 1"
        )
    if not 0.0 < unknown_threshold < 1.0:
        raise ClassificationTrainingError("unknown_threshold must be between 0 and 1")
    if prepare_only and smoke_test:
        raise ClassificationTrainingError(
            "prepare_only and smoke_test cannot be enabled together"
        )

    version = _safe_version(
        model_version or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    run_root = Path(artifact_root) / version
    prepared = prepare_classification_dataset(
        db,
        artifact_root=run_root / "dataset",
        required_leaf_keys=required_leaf_keys,
        minimum_samples_per_leaf=minimum_samples_per_leaf,
    )
    base_result: dict[str, Any] = {
        "status": "prepared" if prepared.report["ready"] else "not_ready",
        "model_family": MODEL_FAMILY,
        "model_version": version,
        "dataset": prepared.report,
        "promotion_threshold": promotion_threshold,
        "unknown_threshold": unknown_threshold,
        "models": [],
        "best_model": None,
        "database_models_created": 0,
        "active_model_changed": False,
        "smoke_test": smoke_test,
    }
    if smoke_test:
        base_result["warning"] = (
            "SMOKE TEST ONLY: this run verifies the training machinery with an "
            "incomplete dataset. It is not a final or production model."
        )
    if prepare_only or (not prepared.report["ready"] and not smoke_test):
        report_path = run_root / "training_report.json"
        base_result["report_path"] = str(report_path.resolve())
        _write_json(report_path, base_result)
        return base_result

    configured_labels = tuple(prepared.report["required_leaf_keys"])
    train_label_counts = Counter(
        item.leaf_key for item in prepared.examples if item.split == "train"
    )
    labels = (
        tuple(
            leaf_key
            for leaf_key in configured_labels
            if train_label_counts[leaf_key] >= 2
        )
        if smoke_test
        else configured_labels
    )
    if len(labels) < 2:
        raise ClassificationTrainingError(
            "smoke test requires at least two labels with two training rows each"
        )
    train_rows = [
        item
        for item in prepared.examples
        if item.split == "train" and item.leaf_key in labels
    ]
    validation_rows = [
        item
        for item in prepared.examples
        if item.split == "validation" and item.leaf_key in labels
    ]
    test_rows = [
        item
        for item in prepared.examples
        if item.split == "test" and item.leaf_key in labels
    ]
    if not validation_rows or not test_rows:
        raise ClassificationTrainingError(
            "training requires non-empty validation and test splits"
        )
    if smoke_test:
        base_result["smoke_test_scope"] = {
            "included_leaf_keys": list(labels),
            "excluded_leaf_keys": [
                leaf_key for leaf_key in configured_labels if leaf_key not in labels
            ],
            "dataset_sample_count": len(train_rows)
            + len(validation_rows)
            + len(test_rows),
            "split_counts": {
                "train": len(train_rows),
                "validation": len(validation_rows),
                "test": len(test_rows),
            },
            "promotion_eligible": False,
        }
    specs = tuple(model_specs or default_classification_model_specs())
    if len(specs) < 2:
        raise ClassificationTrainingError("at least two model specifications are required")

    existing = (
        db.query(ClassificationModel)
        .filter(
            ClassificationModel.model_version == version,
            ClassificationModel.model_key.in_([item.model_key for item in specs]),
        )
        .count()
    )
    if existing:
        raise ClassificationTrainingError(
            f"model_version '{version}' already exists; choose a new version"
        )

    trained_results: list[dict[str, Any]] = []
    estimators: dict[str, Pipeline] = {}
    for spec in specs:
        estimator = spec.factory()
        estimator.fit(
            [item.model_text for item in train_rows],
            [item.leaf_key for item in train_rows],
        )
        validation = _evaluation_by_language(
            estimator,
            validation_rows,
            labels=labels,
            unknown_threshold=unknown_threshold,
        )
        test = _evaluation_by_language(
            estimator,
            test_rows,
            labels=labels,
            unknown_threshold=unknown_threshold,
        )
        gate = _qualification(
            validation[OVERALL_LANGUAGE],
            test[OVERALL_LANGUAGE],
            promotion_threshold=promotion_threshold,
        )
        if smoke_test:
            numeric_gate_passed = bool(gate["passed"])
            gate.update(
                {
                    "numeric_gate_passed": numeric_gate_passed,
                    "passed": False,
                    "eligible_for_promotion": False,
                    "blocked_reason": "smoke_test_incomplete_dataset",
                    "activation": "blocked_smoke_test",
                }
            )
        else:
            gate["numeric_gate_passed"] = bool(gate["passed"])
            gate["eligible_for_promotion"] = bool(gate["passed"])
        trained_results.append(
            {
                "model_key": spec.model_key,
                "model_type": spec.model_type,
                "description": spec.description,
                "validation": validation,
                "test": test,
                "qualification": gate,
            }
        )
        estimators[spec.model_key] = estimator

    ranked = sorted(
        trained_results,
        key=lambda item: (
            -float(item["validation"][OVERALL_LANGUAGE]["f1_macro"]),
            -float(item["validation"][OVERALL_LANGUAGE]["accuracy"]),
            str(item["model_key"]),
        ),
    )
    rank_by_key = {
        str(item["model_key"]): index
        for index, item in enumerate(ranked, start=1)
    }

    dataset_manifest_path = prepared.report["artifacts"].get("manifest_path")
    dataset_manifest_sha256 = prepared.report["artifacts"].get("manifest_sha256")
    now = _utc_now()
    try:
        for result in trained_results:
            model_key = str(result["model_key"])
            model_dir = run_root / model_key
            model_dir.mkdir(parents=True, exist_ok=True)
            artifact_path = model_dir / "model.joblib"
            bundle = {
                "artifact_schema_version": 1,
                "model_family": MODEL_FAMILY,
                "model_key": model_key,
                "model_version": version,
                "model_type": result["model_type"],
                "taxonomy_version": TAXONOMY_VERSION,
                "labels": list(labels),
                "unknown_leaf_key": UNKNOWN_LEAF_KEY,
                "unknown_threshold": unknown_threshold,
                "dataset_fingerprint": prepared.report["dataset_fingerprint"],
                "dataset_manifest_path": dataset_manifest_path,
                "dataset_manifest_sha256": dataset_manifest_sha256,
                "smoke_test_only": smoke_test,
                "training_dataset_ready": bool(prepared.report["ready"]),
                "estimator": estimators[model_key],
            }
            _write_joblib(artifact_path, bundle)
            probe = test_rows[0]
            reloaded_prediction = classify_with_artifact(
                artifact_path,
                title=probe.title,
                text=probe.transcript,
            )
            reload_check = {
                "passed": True,
                "probe_dataset_id": probe.dataset_id,
                "expected_leaf_key": probe.leaf_key,
                "raw_predicted_leaf_key": reloaded_prediction[
                    "raw_taxonomy_leaf_key"
                ],
                "predicted_leaf_key": reloaded_prediction["taxonomy_leaf_key"],
                "confidence": reloaded_prediction["confidence"],
            }
            result["artifact_reload_check"] = reload_check
            evaluation_path = model_dir / "evaluation.json"
            evaluation_payload = {
                key: value
                for key, value in result.items()
                if key not in {"model_id", "artifact_path", "evaluation_path"}
            }
            evaluation_payload.update(
                {
                    "model_version": version,
                    "selection_rank": rank_by_key[model_key],
                    "dataset_fingerprint": prepared.report["dataset_fingerprint"],
                    "artifact_sha256": _sha256_file(artifact_path),
                    "is_active": False,
                }
            )
            _write_json(evaluation_path, evaluation_payload)

            model = ClassificationModel(
                model_key=model_key,
                model_version=version,
                taxonomy_version=TAXONOMY_VERSION,
                model_type=str(result["model_type"]),
                artifact_path=str(artifact_path.resolve()),
                training_dataset_source=YOUTUBE_CC_DATASET_SOURCE,
                training_dataset_version=_dataset_version_for_model(
                    prepared.report
                ),
                training_sample_count=len(train_rows),
                status=(
                    "smoke_test_only"
                    if smoke_test
                    else "qualified"
                    if result["qualification"]["passed"]
                    else "evaluated_below_threshold"
                ),
                is_active=False,
                trained_at=now,
            )
            db.add(model)
            db.flush()
            db.add_all(
                _metric_rows(
                    model_id=model.model_id,
                    dataset_split="validation",
                    evaluations=result["validation"],
                )
            )
            db.add_all(
                _metric_rows(
                    model_id=model.model_id,
                    dataset_split="test",
                    evaluations=result["test"],
                )
            )
            db.add(
                ModelEvaluationMetric(
                    model_id=model.model_id,
                    dataset_split="promotion_gate",
                    language=OVERALL_LANGUAGE,
                    taxonomy_level=3,
                    taxonomy_leaf_key=OVERALL_LEAF_KEY,
                    metric_name="passed",
                    metric_value=(1.0 if result["qualification"]["passed"] else 0.0),
                    sample_size=len(validation_rows) + len(test_rows),
                    details=json.dumps(
                        {
                            **result["qualification"],
                            "selection_rank": rank_by_key[model_key],
                            "unknown_threshold": unknown_threshold,
                            "dataset_fingerprint": prepared.report[
                                "dataset_fingerprint"
                            ],
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                )
            )
            db.add(
                ModelEvaluationMetric(
                    model_id=model.model_id,
                    dataset_split="artifact_check",
                    language=OVERALL_LANGUAGE,
                    taxonomy_level=3,
                    taxonomy_leaf_key=OVERALL_LEAF_KEY,
                    metric_name="reload_classify_passed",
                    metric_value=1.0,
                    sample_size=1,
                    details=json.dumps(
                        reload_check,
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                )
            )
            result["model_id"] = model.model_id
            result["selection_rank"] = rank_by_key[model_key]
            result["artifact_path"] = str(artifact_path.resolve())
            result["artifact_sha256"] = _sha256_file(artifact_path)
            result["evaluation_path"] = str(evaluation_path.resolve())
            result["status"] = model.status
            result["is_active"] = False

        db.add(
            SystemLog(
                action=(
                    "classification_model_smoke_test"
                    if smoke_test
                    else "classification_model_benchmark"
                ),
                status="success",
                detail=json.dumps(
                    {
                        "model_version": version,
                        "dataset_fingerprint": prepared.report[
                            "dataset_fingerprint"
                        ],
                        "models": [
                            {
                                "model_key": item["model_key"],
                                "qualified": item["qualification"]["passed"],
                                "selection_rank": item["selection_rank"],
                            }
                            for item in trained_results
                        ],
                        "active_model_changed": False,
                        "smoke_test": smoke_test,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            )
        )
        db.commit()
    except Exception:
        db.rollback()
        raise

    best = ranked[0]
    base_result.update(
        {
            "status": "smoke_test_evaluated" if smoke_test else "evaluated",
            "models": sorted(
                trained_results, key=lambda item: int(item["selection_rank"])
            ),
            "best_model": {
                "model_key": best["model_key"],
                "model_id": next(
                    item["model_id"]
                    for item in trained_results
                    if item["model_key"] == best["model_key"]
                ),
                "qualified": bool(best["qualification"]["passed"]),
                "smoke_test_only": smoke_test,
                "selection_basis": "validation_macro_f1_then_accuracy",
            },
            "database_models_created": len(trained_results),
            "active_model_changed": False,
        }
    )
    report_path = run_root / "training_report.json"
    base_result["report_path"] = str(report_path.resolve())
    _write_json(report_path, base_result)
    return base_result


def load_classification_artifact(path: str | Path) -> dict[str, Any]:
    payload = joblib.load(Path(path))
    required = {
        "artifact_schema_version",
        "model_key",
        "model_version",
        "labels",
        "unknown_leaf_key",
        "unknown_threshold",
        "estimator",
    }
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ClassificationTrainingError("invalid classification model artifact")
    return payload


def classify_with_artifact(
    path: str | Path,
    *,
    text: str,
    title: str | None = None,
) -> dict[str, Any]:
    payload = load_classification_artifact(path)
    merged_text = f"{title or ''}\n{title or ''}\n{text or ''}".strip()
    raw_prediction = str(payload["estimator"].predict([merged_text])[0])
    predictions, confidences = predict_with_unknown(
        payload["estimator"],
        [merged_text],
        unknown_threshold=float(payload["unknown_threshold"]),
    )
    return {
        "model_key": str(payload["model_key"]),
        "model_version": str(payload["model_version"]),
        "raw_taxonomy_leaf_key": raw_prediction,
        "taxonomy_leaf_key": predictions[0],
        "confidence": confidences[0],
        "is_unknown": predictions[0] == str(payload["unknown_leaf_key"]),
        "smoke_test_only": bool(payload.get("smoke_test_only", False)),
    }
