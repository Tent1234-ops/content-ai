from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.api.deps import require_roles
from app.database.db import get_db
from app.database.models import User
from app.schemas.notifications import NotificationItem, NotificationsResponse, MarkReadRequest
from app.services.notifications import create_notification, get_notifications, mark_notifications_read

router = APIRouter(prefix="/notifications", tags=["notifications"])


@router.get("/", response_model=NotificationsResponse)
def list_notifications(
    unread_only: bool = Query(default=False),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    current_user: User = Depends(require_roles("user", "admin")),
    db: Session = Depends(get_db),
):
    res = get_notifications(db=db, user_id=current_user.user_id, unread_only=unread_only, limit=limit, offset=offset)
    return NotificationsResponse(total=res["total"], items=res["items"])


@router.post("/mark_read")
def mark_read(
    request: MarkReadRequest,
    current_user: User = Depends(require_roles("user", "admin")),
    db: Session = Depends(get_db),
):
    marked = mark_notifications_read(db=db, user_id=current_user.user_id, ids=request.ids)
    return {"marked": marked}


# Admin helper to create a notification for a user (useful for testing or backfill)
@router.post("/create_for_user", response_model=NotificationItem)
def create_for_user(
    user_id: int,
    title: str,
    body: str | None = None,
    current_user: User = Depends(require_roles("admin")),
    db: Session = Depends(get_db),
):
    n = create_notification(db=db, user_id=user_id, title=title, body=body)
    return n
