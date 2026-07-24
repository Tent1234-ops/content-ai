from typing import Dict, List
from sqlalchemy.orm import Session
from app.services.trends import get_youtube_trending, get_google_trending
from app.services.follows import get_subscribers_for_value
from app.services.notifications import create_notification
from app.database.models import FollowedTopic, Notification
from app.core.config import settings


def _normalize_text(s: str) -> str:
    return (s or "").strip().lower()


def scan_youtube_trends(db: Session, *, region: str | None = None, limit: int = 20, mode: str = "auto") -> Dict[str, int]:
    """Scan YouTube trending items and create notifications for followed domains/keywords.
    Returns stats: {'items_processed': int, 'notifications_created': int}
    """
    stats = {"items_processed": 0, "notifications_created": 0}
    region = (region or settings.youtube_region).upper()
    resolved_mode, items = get_youtube_trending(region=region, limit=limit, mode=mode)

    # Preload distinct followed keywords (lowercased) to check substring matches
    kw_rows = db.query(FollowedTopic.value).filter(FollowedTopic.match_type == "keyword").distinct().all()
    followed_keywords = [r[0] for r in kw_rows]

    for item in items:
        stats["items_processed"] += 1
        title = getattr(item, "title", "")
        # attempt to find a canonical link for dedupe (video_url or url)
        link = getattr(item, "video_url", None) or getattr(item, "url", None)
        domain = getattr(item, "category", None) or "general"

        # Domain matches
        domain_subs = get_subscribers_for_value(db=db, match_type="domain", value=_normalize_text(domain))
        for user_id in domain_subs:
            # dedupe by link if available
            if link:
                exists = db.query(Notification).filter(Notification.user_id == user_id, Notification.link == link).first()
                if exists:
                    continue
            create_notification(
                db=db,
                user_id=user_id,
                title=f"New trend in {domain}: {title[:120]}",
                body=getattr(item, "description", None) or title,
                link=link,
                type="trend",
                payload={"source": "youtube", "domain": domain, "trend_id": getattr(item, "video_url", None) or getattr(item, "id", None)},
            )
            stats["notifications_created"] += 1

        # Keyword substring matches (cheap approach)
        text_to_search = _normalize_text(" ".join([title, getattr(item, "transcript", "") or "", getattr(item, "description", "") or ""]))
        if not text_to_search:
            continue
        for kw in followed_keywords:
            if not kw:
                continue
            if kw in text_to_search:
                subs = get_subscribers_for_value(db=db, match_type="keyword", value=kw)
                for user_id in subs:
                    if link:
                        exists = db.query(Notification).filter(Notification.user_id == user_id, Notification.link == link).first()
                        if exists:
                            continue
                    create_notification(
                        db=db,
                        user_id=user_id,
                        title=f"New {kw} trend: {title[:120]}",
                        body=getattr(item, "description", None) or title,
                        link=link,
                        type="trend_keyword",
                        payload={"source": "youtube", "keyword": kw, "trend_id": getattr(item, "video_url", None) or getattr(item, "id", None)},
                    )
                    stats["notifications_created"] += 1

    return stats


def scan_google_trends(db: Session, *, region: str | None = None, limit: int = 20, mode: str = "auto") -> Dict[str, int]:
    stats = {"items_processed": 0, "notifications_created": 0}
    region = (region or settings.youtube_region).upper()
    resolved_mode, items = get_google_trending(region=region, limit=limit, mode=mode)

    kw_rows = db.query(FollowedTopic.value).filter(FollowedTopic.match_type == "keyword").distinct().all()
    followed_keywords = [r[0] for r in kw_rows]

    for item in items:
        stats["items_processed"] += 1
        title = getattr(item, "title", "")
        link = getattr(item, "url", None)
        domain = getattr(item, "category", None) or "general"

        domain_subs = get_subscribers_for_value(db=db, match_type="domain", value=_normalize_text(domain))
        for user_id in domain_subs:
            if link:
                exists = db.query(Notification).filter(Notification.user_id == user_id, Notification.link == link).first()
                if exists:
                    continue
            create_notification(
                db=db,
                user_id=user_id,
                title=f"New trend in {domain}: {title[:120]}",
                body=getattr(item, "snippet", None) or title,
                link=link,
                type="trend",
                payload={"source": "google", "domain": domain, "trend_id": getattr(item, "id", None)},
            )
            stats["notifications_created"] += 1

        text_to_search = _normalize_text(" ".join([title, getattr(item, "snippet", "") or ""]))
        if not text_to_search:
            continue
        for kw in followed_keywords:
            if not kw:
                continue
            if kw in text_to_search:
                subs = get_subscribers_for_value(db=db, match_type="keyword", value=kw)
                for user_id in subs:
                    if link:
                        exists = db.query(Notification).filter(Notification.user_id == user_id, Notification.link == link).first()
                        if exists:
                            continue
                    create_notification(
                        db=db,
                        user_id=user_id,
                        title=f"New {kw} trend: {title[:120]}",
                        body=getattr(item, "snippet", None) or title,
                        link=link,
                        type="trend_keyword",
                        payload={"source": "google", "keyword": kw, "trend_id": getattr(item, "id", None)},
                    )
                    stats["notifications_created"] += 1

    return stats
