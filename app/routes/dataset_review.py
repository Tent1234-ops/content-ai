from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.api.deps import require_roles
from app.database.db import get_db
from app.database.models import SystemLog, User
from app.schemas.dataset_review import (
    DatasetReviewDecisionRequest,
    DatasetReviewDecisionResponse,
    DatasetReviewQueueResponse,
)
from app.services.youtube_cc_dataset import (
    YouTubeCCDatasetError,
    list_youtube_cc_review_queue,
    review_youtube_cc_candidate,
)


router = APIRouter(prefix="/admin/dataset-review", tags=["admin-dataset-review"])


@router.get("/queue", response_model=DatasetReviewQueueResponse)
def dataset_review_queue(
    limit: int = Query(default=12, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    status: str = Query(default="pending", pattern="^(pending|approved|rejected|all)$"),
    leaf_key: str | None = Query(default=None),
    collection_run_id: int | None = Query(default=None, ge=1),
    search: str | None = Query(default=None, max_length=200),
    _current_user: User = Depends(require_roles("admin")),
    db: Session = Depends(get_db),
):
    try:
        return list_youtube_cc_review_queue(
            db,
            limit=limit,
            offset=offset,
            review_status=status,
            leaf_key=leaf_key,
            collection_run_id=collection_run_id,
            search=search,
        )
    except YouTubeCCDatasetError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post(
    "/runs/{collection_run_id}/candidates/{source_youtube_id}",
    response_model=DatasetReviewDecisionResponse,
)
def review_dataset_candidate(
    collection_run_id: int,
    source_youtube_id: str,
    payload: DatasetReviewDecisionRequest,
    current_user: User = Depends(require_roles("admin")),
    db: Session = Depends(get_db),
):
    reviewer = f"{current_user.username} <{current_user.email}> [user:{current_user.user_id}]"
    try:
        result = review_youtube_cc_candidate(
            db,
            collection_run_id=collection_run_id,
            source_youtube_id=source_youtube_id,
            decision=payload.decision,
            reviewer=reviewer,
            reviewed_leaf_key=payload.reviewed_leaf_key,
            transcript_quality=payload.transcript_quality,
            notes=payload.notes,
        )
    except YouTubeCCDatasetError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    db.add(
        SystemLog(
            user_id=current_user.user_id,
            action=f"dataset_review_{payload.decision}",
            status="success",
            detail=(
                f"YouTube CC candidate {source_youtube_id}; "
                f"run={collection_run_id}; leaf={payload.reviewed_leaf_key or '-'}"
            ),
        )
    )
    db.commit()
    return result
