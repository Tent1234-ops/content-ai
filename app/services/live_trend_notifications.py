from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from typing import Dict, Iterable, List

from sqlalchemy.orm import Session

from app.database.models import (
    Notification,
    TrendSnapshotItem,
    TrendSnapshotRun,
    User,
    UserTrendWatchSession,
)
from app.services.live_trend_snapshots import PLATFORMS, load_latest_live_snapshot
from app.services.notifications import create_live_trend_notification


SUCCESS_PROVIDER_STATES = {"ok", "empty"}


def _provider_states(run: TrendSnapshotRun) -> Dict[str, str]:
    try:
        payload = json.loads(run.provider_status or "{}")
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    return {
        platform: str(details.get("status") or "error")
        for platform, details in payload.items()
        if isinstance(details, dict)
    }


def _items_by_platform(db: Session, *, run_id: int) -> Dict[str, List[TrendSnapshotItem]]:
    grouped: Dict[str, List[TrendSnapshotItem]] = defaultdict(list)
    rows = (
        db.query(TrendSnapshotItem)
        .filter(TrendSnapshotItem.run_id == run_id)
        .order_by(TrendSnapshotItem.platform, TrendSnapshotItem.item_id)
        .all()
    )
    for row in rows:
        grouped[row.platform].append(row)
    return grouped


def _latest_successful_items_before(
    db: Session,
    *,
    region: str,
    platform: str,
    run_id: int,
) -> List[TrendSnapshotItem] | None:
    runs = (
        db.query(TrendSnapshotRun)
        .filter(
            TrendSnapshotRun.region == region,
            TrendSnapshotRun.run_id <= run_id,
            TrendSnapshotRun.status.in_(("completed", "partial")),
        )
        .order_by(TrendSnapshotRun.run_id.desc())
        .limit(120)
        .all()
    )
    for run in runs:
        if _provider_states(run).get(platform) not in SUCCESS_PROVIDER_STATES:
            continue
        return (
            db.query(TrendSnapshotItem)
            .filter(
                TrendSnapshotItem.run_id == run.run_id,
                TrendSnapshotItem.platform == platform,
            )
            .all()
        )
    return None


def _apply_session_and_engagement_status(
    db: Session,
    *,
    snapshot: Dict[str, object],
    watch_session: UserTrendWatchSession,
) -> None:
    notified_keys = {
        key
        for (key,) in (
            db.query(Notification.trend_key)
            .filter(Notification.watch_session_id == watch_session.watch_session_id)
            .all()
        )
    }
    run_id = snapshot.get("run_id")
    if not isinstance(run_id, int):
        return

    region = str(snapshot.get("region") or "TH")
    for platform in PLATFORMS:
        platform_payload = snapshot.get("platforms", {}).get(platform, {})
        items = platform_payload.get("items", [])
        previous_rows = _latest_successful_items_before(
            db,
            region=region,
            platform=platform,
            run_id=run_id - 1,
        )
        previous_signals = {
            row.trend_key: float(row.engagement_signal or 0.0)
            for row in (previous_rows or [])
        }
        max_signal = max(
            (float(item.get("engagement_signal") or 0.0) for item in items),
            default=0.0,
        )
        for item in items:
            key = str(item.get("key") or "")
            current = float(item.get("engagement_signal") or 0.0)
            previous = previous_signals.get(key)
            change = ((current - previous) / previous) * 100.0 if previous and previous > 0 else 0.0
            is_new = key in notified_keys
            if is_new or change >= 15:
                status = "Rising"
            elif change <= -10:
                status = "Cooling"
            elif max_signal > 0 and current >= max_signal * 0.75:
                status = "Hot"
            else:
                status = "Stable"
            item["engagement_change_percent"] = round(change, 1)
            item["status"] = status
            item["is_new"] = is_new


def compare_live_trend_snapshot(
    db: Session,
    *,
    user: User,
    watch_session: UserTrendWatchSession,
    region: str,
    limit: int,
    platforms: Iterable[str] = PLATFORMS,
) -> Dict[str, object]:
    region = region.upper()
    platforms = tuple(platforms)
    snapshot = load_latest_live_snapshot(db, region=region, limit=limit)
    latest_run = (
        db.query(TrendSnapshotRun)
        .filter(
            TrendSnapshotRun.region == region,
            TrendSnapshotRun.status.in_(("completed", "partial")),
        )
        .order_by(TrendSnapshotRun.run_id.desc())
        .first()
    )
    new_notifications: List[Notification] = []

    if latest_run is None:
        _apply_session_and_engagement_status(db, snapshot=snapshot, watch_session=watch_session)
        return {
            **snapshot,
            "new_count": 0,
            "new_notifications": [],
        }

    if watch_session.baseline_run_id is None or watch_session.last_seen_run_id is None:
        watch_session.baseline_run_id = latest_run.run_id
        watch_session.last_seen_run_id = latest_run.run_id
        watch_session.last_seen_at = datetime.utcnow()
        db.commit()
        _apply_session_and_engagement_status(db, snapshot=snapshot, watch_session=watch_session)
        return {
            **snapshot,
            "new_count": 0,
            "new_notifications": [],
        }

    last_seen_run_id = int(watch_session.last_seen_run_id)
    if latest_run.run_id > last_seen_run_id:
        known_keys: Dict[str, set[str] | None] = {}
        for platform in platforms:
            previous_items = _latest_successful_items_before(
                db,
                region=region,
                platform=platform,
                run_id=last_seen_run_id,
            )
            known_keys[platform] = (
                {item.trend_key for item in previous_items}
                if previous_items is not None
                else None
            )

        runs = (
            db.query(TrendSnapshotRun)
            .filter(
                TrendSnapshotRun.region == region,
                TrendSnapshotRun.run_id > last_seen_run_id,
                TrendSnapshotRun.run_id <= latest_run.run_id,
                TrendSnapshotRun.status.in_(("completed", "partial")),
            )
            .order_by(TrendSnapshotRun.run_id)
            .all()
        )
        for run in runs:
            states = _provider_states(run)
            grouped_items = _items_by_platform(db, run_id=run.run_id)
            detected_at = run.completed_at or datetime.utcnow()
            for platform in platforms:
                if states.get(platform) not in SUCCESS_PROVIDER_STATES:
                    continue
                current_items = grouped_items.get(platform, [])
                current_keys = {item.trend_key for item in current_items}
                previous_keys = known_keys.get(platform)
                if previous_keys is not None:
                    for item in current_items:
                        if item.trend_key in previous_keys:
                            continue
                        notification = create_live_trend_notification(
                            db,
                            user_id=user.user_id,
                            watch_session_id=watch_session.watch_session_id,
                            item=item,
                            detected_at=detected_at,
                        )
                        if notification is not None:
                            new_notifications.append(notification)
                known_keys[platform] = current_keys

        watch_session.last_seen_run_id = latest_run.run_id
        watch_session.last_seen_at = datetime.utcnow()
        db.commit()

    _apply_session_and_engagement_status(db, snapshot=snapshot, watch_session=watch_session)
    return {
        **snapshot,
        "new_count": len(new_notifications),
        "new_notifications": new_notifications,
    }
