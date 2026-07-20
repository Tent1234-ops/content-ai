from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import require_roles
from app.database.db import get_db
from app.database.models import User
from app.schemas.classification import ClassificationRequest, ClassificationResponse
from app.services.classification import classify_text_domain

router = APIRouter(prefix="/classification", tags=["classification"])


@router.post("/text", response_model=ClassificationResponse)
def classify_text(
    payload: ClassificationRequest,
    _current_user: User = Depends(require_roles("admin", "user")),
    db: Session = Depends(get_db),
):
    return ClassificationResponse.model_validate(
        classify_text_domain(
            db,
            title=payload.title,
            text=payload.text,
            source_prefix=payload.source,
            profile_limit=payload.profile_limit,
            top_k=payload.top_k,
        )
    )
