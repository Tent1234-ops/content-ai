from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from app.database.db import Base, SessionLocal, engine
from app.database.migrations import (
    migrate_phase13_taxonomy_schema,
    migrate_youtube_cc_dataset_schema,
)
from app.services.taxonomy import sync_taxonomy_registry, taxonomy_coverage


def main() -> int:
    Base.metadata.create_all(bind=engine)
    phase13 = migrate_phase13_taxonomy_schema(engine)
    youtube_cc = migrate_youtube_cc_dataset_schema(engine)
    db = SessionLocal()
    try:
        taxonomy = sync_taxonomy_registry(db)
        coverage = taxonomy_coverage(db)
    finally:
        db.close()
    print(
        json.dumps(
            {
                "phase13_schema": phase13,
                "youtube_cc_schema": youtube_cc,
                "taxonomy_sync": taxonomy,
                "coverage": coverage,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
