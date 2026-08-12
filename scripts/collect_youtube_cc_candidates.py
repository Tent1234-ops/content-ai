from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from app.core.config import settings
from app.database.db import Base, SessionLocal, engine
from app.database.migrations import (
    migrate_phase13_taxonomy_schema,
    migrate_youtube_cc_dataset_schema,
)
from app.services.dataset_contract import DEFAULT_YOUTUBE_CC_DATASET_VERSION
from app.services.taxonomy import ACTIVE_LEAF_KEYS, sync_taxonomy_registry
from app.services.youtube_cc_dataset import (
    YouTubeCCDatasetError,
    collect_youtube_cc_candidates,
)


def _csv_values(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect real YouTube Creative Commons videos and public captions. "
            "The generated review CSV contains no automatic approvals."
        )
    )
    parser.add_argument("--dataset-version", default=DEFAULT_YOUTUBE_CC_DATASET_VERSION)
    parser.add_argument(
        "--leaves",
        default=",".join(ACTIVE_LEAF_KEYS),
        help="Comma-separated taxonomy leaves",
    )
    parser.add_argument("--languages", default="th,en")
    parser.add_argument("--target-per-leaf", type=int, default=50)
    parser.add_argument("--region", default=settings.youtube_region)
    parser.add_argument("--max-pages-per-query", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional output directory; defaults to data/raw/youtube_cc/<version>",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not settings.youtube_api_key:
        print("ERROR: YOUTUBE_API_KEY is not configured in .env", file=sys.stderr)
        return 1

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    raw_dir = args.output_dir or ROOT / "data" / "raw" / "youtube_cc" / args.dataset_version
    candidate_path = raw_dir / f"candidates-{stamp}.jsonl"
    review_path = ROOT / "data" / "reviews" / "youtube_cc" / args.dataset_version / f"review-{stamp}.csv"
    manifest_path = ROOT / "data" / "manifests" / f"youtube-cc-{args.dataset_version}-{stamp}.json"

    Base.metadata.create_all(bind=engine)
    migrate_phase13_taxonomy_schema(engine)
    migrate_youtube_cc_dataset_schema(engine)
    db = SessionLocal()
    try:
        sync_taxonomy_registry(db)
        result = collect_youtube_cc_candidates(
            db,
            api_key=settings.youtube_api_key,
            candidate_path=candidate_path,
            review_path=review_path,
            manifest_path=manifest_path,
            dataset_version=args.dataset_version,
            leaf_keys=_csv_values(args.leaves),
            target_per_leaf=args.target_per_leaf,
            languages=_csv_values(args.languages),
            region_code=args.region,
            max_pages_per_query=args.max_pages_per_query,
            timeout_seconds=args.timeout,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"\nHuman review file: {review_path}")
        print("Fill decision/reviewed_leaf_key/quality/reviewer/reviewed_at before import.")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except YouTubeCCDatasetError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
