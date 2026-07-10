from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.api.deps import require_roles
from app.database.db import get_db
from app.database.models import User
from app.schemas.admin_report import (
    AdminClusterRunListResponse,
    AdminClusterRunDetailResponse,
    AdminDatasetItem,
    AdminDatasetListResponse,
    AdminOverviewReportResponse,
    AdminSystemLogItem,
    AdminSystemLogListResponse,
)
from app.schemas.auth import UserResponse
from app.services.admin_report import (
    build_admin_overview_report,
    get_admin_cluster_run_detail,
    list_admin_cluster_runs,
    list_admin_datasets,
    list_admin_logs,
)

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/me")
def admin_profile(current_user: User = Depends(require_roles("admin"))):
    return {
        "message": "Admin access granted",
        "user": UserResponse.model_validate(current_user),
    }


@router.get("/datasets", response_model=AdminDatasetListResponse)
def admin_datasets(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    source: str | None = Query(default=None),
    category: str | None = Query(default=None),
    search: str | None = Query(default=None),
    _current_user: User = Depends(require_roles("admin")),
    db: Session = Depends(get_db),
):
    total, items = list_admin_datasets(
        db,
        limit=limit,
        offset=offset,
        source=source,
        category=category,
        search=search,
    )
    return AdminDatasetListResponse(
        total=total,
        items=[AdminDatasetItem.model_validate(item, from_attributes=True) for item in items],
    )


@router.get("/clusters/runs", response_model=AdminClusterRunListResponse)
def admin_cluster_runs(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    algorithm: str | None = Query(default=None),
    _current_user: User = Depends(require_roles("admin")),
    db: Session = Depends(get_db),
):
    total, items = list_admin_cluster_runs(db, limit=limit, offset=offset, algorithm=algorithm)
    return AdminClusterRunListResponse(total=total, items=items)


@router.get("/clusters/runs/{run_id}", response_model=AdminClusterRunDetailResponse)
def admin_cluster_run_detail(
    run_id: int,
    _current_user: User = Depends(require_roles("admin")),
    db: Session = Depends(get_db),
):
    item = get_admin_cluster_run_detail(db, run_id=run_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Cluster run not found")
    return AdminClusterRunDetailResponse.model_validate(item)


@router.get("/logs", response_model=AdminSystemLogListResponse)
def admin_logs(
    limit: int = Query(default=30, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    status: str | None = Query(default=None),
    action: str | None = Query(default=None),
    _current_user: User = Depends(require_roles("admin")),
    db: Session = Depends(get_db),
):
    total, items = list_admin_logs(db, limit=limit, offset=offset, status=status, action=action)
    return AdminSystemLogListResponse(
        total=total,
        items=[AdminSystemLogItem.model_validate(item, from_attributes=True) for item in items],
    )


@router.get("/reports/overview", response_model=AdminOverviewReportResponse)
def admin_reports_overview(
    _current_user: User = Depends(require_roles("admin")),
    db: Session = Depends(get_db),
):
    return AdminOverviewReportResponse.model_validate(build_admin_overview_report(db))
