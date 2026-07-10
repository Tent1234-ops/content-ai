from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.database.db import get_db
from app.database.models import User
from app.schemas.contents import (
    UserContentDetailResponse,
    UserContentHistoryResponse,
)
from app.services.contents import get_user_content_detail, list_user_contents

router = APIRouter(prefix="/contents", tags=["contents"])


@router.get("/my", response_model=UserContentHistoryResponse)
def my_contents(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    total, items = list_user_contents(db, user_id=current_user.user_id, limit=limit, offset=offset)
    return UserContentHistoryResponse(total=total, items=items)


@router.get("/{content_id}", response_model=UserContentDetailResponse)
def content_detail(
    content_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    item = get_user_content_detail(db, user_id=current_user.user_id, content_id=content_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Content not found")
    return UserContentDetailResponse.model_validate(item)
