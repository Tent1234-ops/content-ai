from sqlalchemy import func
from sqlalchemy.orm import Session

from app.database.models import Cluster, ClusterMembership, ClusterRun, DatasetContent, SystemLog
from app.schemas.admin_report import AdminDatasetCreate, AdminDatasetUpdate
from app.services.persistence import log_system_event


def list_admin_datasets(
    db: Session,
    *,
    limit: int = 20,
    offset: int = 0,
    source: str | None = None,
    category: str | None = None,
    search: str | None = None,
):
    query = db.query(DatasetContent)
    if source:
        query = query.filter(DatasetContent.source_platform.like(f"{source}%"))
    if category:
        query = query.filter(DatasetContent.category == category)
    if search:
        like_value = f"%{search.strip()}%"
        query = query.filter(
            (DatasetContent.title.like(like_value)) | (DatasetContent.transcript.like(like_value))
        )
    total = query.count()
    items = (
        query.order_by(DatasetContent.created_at.desc(), DatasetContent.trend_score.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return total, items


def create_admin_dataset(
    db: Session,
    *,
    payload: AdminDatasetCreate,
    user_id: int | None = None,
) -> DatasetContent:
    item = DatasetContent(**payload.model_dump())
    db.add(item)
    db.flush()
    log_system_event(
        db,
        user_id=user_id,
        action="admin_dataset_create",
        status="success",
        detail=f"dataset_id={item.dataset_id}, source={item.source_platform}",
    )
    db.commit()
    db.refresh(item)
    return item


def update_admin_dataset(
    db: Session,
    *,
    dataset_id: int,
    payload: AdminDatasetUpdate,
    user_id: int | None = None,
) -> DatasetContent | None:
    item = db.query(DatasetContent).filter(DatasetContent.dataset_id == dataset_id).first()
    if item is None:
        return None
    update_data = payload.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(item, key, value)
    log_system_event(
        db,
        user_id=user_id,
        action="admin_dataset_update",
        status="success",
        detail=f"dataset_id={dataset_id}, fields={list(update_data.keys())}",
    )
    db.commit()
    db.refresh(item)
    return item


def list_admin_cluster_runs(
    db: Session,
    *,
    limit: int = 20,
    offset: int = 0,
    algorithm: str | None = None,
):
    query = db.query(ClusterRun)
    if algorithm:
        query = query.filter(ClusterRun.algorithm == algorithm)
    total = query.count()
    rows = (
        query.with_entities(ClusterRun, func.count(ClusterMembership.membership_id))
        .outerjoin(ClusterMembership, ClusterMembership.run_id == ClusterRun.run_id)
        .group_by(ClusterRun.run_id)
        .order_by(ClusterRun.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    items = [
        {
            "run_id": run.run_id,
            "algorithm": run.algorithm,
            "n_clusters": run.n_clusters,
            "feature_dimension": run.feature_dimension,
            "inertia": float(run.inertia),
            "membership_count": int(membership_count),
            "created_at": run.created_at,
        }
        for run, membership_count in rows
    ]
    return total, items


def list_admin_logs(
    db: Session,
    *,
    limit: int = 30,
    offset: int = 0,
    status: str | None = None,
    action: str | None = None,
):
    query = db.query(SystemLog)
    if status:
        query = query.filter(SystemLog.status == status)
    if action:
        query = query.filter(SystemLog.action.like(f"%{action.strip()}%"))
    total = query.count()
    items = query.order_by(SystemLog.timestamp.desc()).offset(offset).limit(limit).all()
    return total, items


def build_admin_overview_report(db: Session):
    dataset_total = db.query(func.count(DatasetContent.dataset_id)).scalar() or 0
    cluster_run_total = db.query(func.count(ClusterRun.run_id)).scalar() or 0
    system_log_total = db.query(func.count(SystemLog.log_id)).scalar() or 0

    top_sources_raw = (
        db.query(
            DatasetContent.source_platform,
            func.count(DatasetContent.dataset_id),
            func.avg(DatasetContent.trend_score),
        )
        .group_by(DatasetContent.source_platform)
        .order_by(func.count(DatasetContent.dataset_id).desc())
        .limit(10)
        .all()
    )
    top_sources = [
        {
            "source_platform": source_platform or "unknown",
            "count": int(count),
            "avg_trend_score": round(float(avg_trend_score or 0.0), 3),
        }
        for source_platform, count, avg_trend_score in top_sources_raw
    ]

    top_categories_raw = (
        db.query(DatasetContent.category, func.count(DatasetContent.dataset_id))
        .group_by(DatasetContent.category)
        .order_by(func.count(DatasetContent.dataset_id).desc())
        .limit(10)
        .all()
    )
    top_categories = [
        {"category": category or "uncategorized", "count": int(count)}
        for category, count in top_categories_raw
    ]

    status_breakdown_raw = (
        db.query(SystemLog.status, func.count(SystemLog.log_id))
        .group_by(SystemLog.status)
        .order_by(func.count(SystemLog.log_id).desc())
        .all()
    )
    status_breakdown = [
        {"status": status or "unknown", "count": int(count)}
        for status, count in status_breakdown_raw
    ]

    recent_actions_raw = db.query(SystemLog).order_by(SystemLog.timestamp.desc()).limit(10).all()
    recent_actions = [
        {
            "log_id": log.log_id,
            "action": log.action,
            "status": log.status,
            "timestamp": log.timestamp.isoformat() if log.timestamp else None,
            "detail": log.detail or "",
        }
        for log in recent_actions_raw
    ]

    return {
        "dataset_total": int(dataset_total),
        "cluster_run_total": int(cluster_run_total),
        "system_log_total": int(system_log_total),
        "top_sources": top_sources,
        "top_categories": top_categories,
        "status_breakdown": status_breakdown,
        "recent_actions": recent_actions,
    }


def get_admin_cluster_run_detail(db: Session, *, run_id: int):
    run = db.query(ClusterRun).filter(ClusterRun.run_id == run_id).first()
    if run is None:
        return None

    cluster_rows = (
        db.query(
            ClusterMembership.cluster_id,
            func.count(ClusterMembership.membership_id),
            func.max(ClusterMembership.created_at),
        )
        .filter(ClusterMembership.run_id == run_id)
        .group_by(ClusterMembership.cluster_id)
        .all()
    )
    cluster_names = {
        cluster_id: cluster_name
        for cluster_id, cluster_name in (
            db.query(Cluster.cluster_id, Cluster.cluster_name)
            .join(ClusterMembership, ClusterMembership.cluster_id == Cluster.cluster_id)
            .filter(ClusterMembership.run_id == run_id)
            .group_by(Cluster.cluster_id, Cluster.cluster_name)
            .all()
        )
    }

    cluster_breakdown = [
        {
            "cluster_id": int(cluster_id),
            "cluster_name": cluster_names.get(cluster_id) or f"cluster_{cluster_id}",
            "member_count": int(member_count),
        }
        for cluster_id, member_count, _last_seen in cluster_rows
    ]
    cluster_breakdown.sort(key=lambda item: (-item["member_count"], item["cluster_name"]))

    recent_membership_rows = (
        db.query(ClusterMembership)
        .join(ClusterMembership.cluster)
        .filter(ClusterMembership.run_id == run_id)
        .order_by(ClusterMembership.created_at.desc())
        .limit(20)
        .all()
    )
    recent_memberships = [
        {
            "membership_id": row.membership_id,
            "cluster_id": row.cluster_id,
            "cluster_name": row.cluster.cluster_name if row.cluster else f"cluster_{row.cluster_id}",
            "dataset_id": row.dataset_id,
            "content_id": row.content_id,
            "item_text": row.item_text,
            "top_terms": row.top_terms,
            "created_at": row.created_at,
        }
        for row in recent_membership_rows
    ]

    return {
        "run_id": run.run_id,
        "algorithm": run.algorithm,
        "n_clusters": run.n_clusters,
        "feature_dimension": run.feature_dimension,
        "inertia": float(run.inertia or 0.0),
        "membership_count": sum(item["member_count"] for item in cluster_breakdown),
        "created_at": run.created_at,
        "cluster_breakdown": cluster_breakdown,
        "recent_memberships": recent_memberships,
    }
