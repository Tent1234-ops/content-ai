from typing import Dict, List, Optional
from sqlalchemy.orm import Session
import json

from app.database.models import Notification


def create_notification(
    db: Session,
    *,
    user_id: int,
    title: str,
    body: Optional[str] = None,
    link: Optional[str] = None,
    type: str = "system",
    payload: Optional[dict] = None,
    delivered_via_ws: bool = False,
) -> Notification:
    payload_text = json.dumps(payload, ensure_ascii=False) if payload is not None else None
    n = Notification(
        user_id=user_id,
        title=title[:255],
        body=body,
        link=link,
        type=type,
        payload=payload_text,
        is_read=False,
        delivered_via_ws=delivered_via_ws,
    )
    db.add(n)
    db.flush()
    db.commit()
    return n


def get_notifications(
    db: Session,
    *,
    user_id: int,
    unread_only: bool = False,
    limit: int = 50,
    offset: int = 0,
) -> Dict[str, object]:
    q = db.query(Notification).filter(Notification.user_id == user_id)
    if unread_only:
        q = q.filter(Notification.is_read == False)
    total = q.count()
    items = q.order_by(Notification.created_at.desc()).offset(offset).limit(limit).all()
    return {"total": total, "items": items}


def mark_notifications_read(db: Session, *, user_id: int, ids: List[int]) -> int:
    if not ids:
        return 0
    rows = db.query(Notification).filter(Notification.user_id == user_id, Notification.notification_id.in_(ids)).all()
    for r in rows:
        r.is_read = True
    db.commit()
    return len(rows)
