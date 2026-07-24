from typing import Dict, List, Optional
from sqlalchemy.orm import Session
from app.database.models import FollowedTopic
from datetime import datetime


def follow_topic(db: Session, *, user_id: int, match_type: str, value: str) -> FollowedTopic:
    norm_value = (value or "").strip().lower()
    match_type = (match_type or "").strip().lower()
    if match_type not in ("domain", "keyword"):
        raise ValueError("match_type must be 'domain' or 'keyword'")
    existing = (
        db.query(FollowedTopic)
        .filter(FollowedTopic.user_id == user_id, FollowedTopic.match_type == match_type, FollowedTopic.value == norm_value)
        .first()
    )
    if existing:
        return existing
    ft = FollowedTopic(user_id=user_id, match_type=match_type, value=norm_value, created_at=datetime.utcnow())
    db.add(ft)
    db.flush()
    db.commit()
    return ft


def unfollow_topic(db: Session, *, user_id: int, id: Optional[int] = None, value: Optional[str] = None) -> int:
    if id is None and value is None:
        return 0
    q = db.query(FollowedTopic).filter(FollowedTopic.user_id == user_id)
    if id is not None:
        q = q.filter(FollowedTopic.id == int(id))
    if value is not None:
        q = q.filter(FollowedTopic.value == (value or "").strip().lower())
    rows = q.all()
    deleted = 0
    for r in rows:
        db.delete(r)
        deleted += 1
    if deleted:
        db.commit()
    return deleted


def list_followed_topics(db: Session, *, user_id: int, limit: int = 100, offset: int = 0) -> Dict[str, object]:
    q = db.query(FollowedTopic).filter(FollowedTopic.user_id == user_id)
    total = q.count()
    items = q.order_by(FollowedTopic.created_at.desc()).offset(offset).limit(limit).all()
    return {"total": total, "items": items}


def get_subscribers_for_value(db: Session, *, match_type: str, value: str) -> List[int]:
    # returns list of user_ids following the given match
    norm_value = (value or "").strip().lower()
    q = db.query(FollowedTopic).filter(FollowedTopic.match_type == match_type, FollowedTopic.value == norm_value)
    return [row.user_id for row in q.all()]
