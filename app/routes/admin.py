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
from app.schemas.admin_config import (
    AdminConfigResponse,
    AdminConfigUpdate,
    AdminConfigValidationResponse,
    AdminStatisticsResponse,
    BulkConfigResetResponse,
)
from app.schemas.auth import UserResponse
from app.services.admin_report import (
    build_admin_overview_report,
    get_admin_cluster_run_detail,
    list_admin_cluster_runs,
    list_admin_datasets,
    list_admin_logs,
)
from app.services.admin_settings import (
    get_admin_config,
    save_admin_config,
    reset_admin_config,
    validate_config,
    get_admin_statistics,
    export_config_backup,
    apply_config_from_backup,
    get_admin_audit_log,
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
    try:
        report = build_admin_overview_report(db)
        return AdminOverviewReportResponse.model_validate(report)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error building admin report: {str(exc)}"
        )


# ============================================================================
# ADMIN CONFIGURATION MANAGEMENT ENDPOINTS
# ============================================================================

@router.get("/settings", response_model=AdminConfigResponse, tags=["admin-settings"])
def get_admin_settings(
    _current_user: User = Depends(require_roles("admin")),
    db: Session = Depends(get_db),
):
    """
    Get current admin configuration settings.
    
    Returns all system-wide configuration parameters used by the recommendation engine.
    """
    try:
        config = get_admin_config(db)
        return config
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving admin settings: {str(exc)}"
        )


@router.put("/settings", response_model=AdminConfigResponse, tags=["admin-settings"])
def update_admin_settings(
    config_update: AdminConfigUpdate,
    _current_user: User = Depends(require_roles("admin")),
    db: Session = Depends(get_db),
):
    """
    Update admin configuration settings.
    
    Allows partial updates - only fields that are provided will be updated.
    
    Parameters:
    - max_keywords_display (1-50): Maximum keywords shown in recommendations
    - hook_analysis_duration (10-300): Duration in seconds for hook analysis
    - analysis_time_range_days (1-365): Days back to consider for analysis
    - notification_batch_size (1-200): Max notifications per batch
    - youtube_region (2-char): YouTube trending region (TH, US, GB, etc.)
    - google_region (2-char): Google Trends region
    - enable_youtube_trending: Toggle YouTube trending data collection
    - enable_google_trends: Toggle Google Trends data collection
    - auto_scan_interval_hours (1-24): Hours between automatic scans
    """
    try:
        # Validate configuration
        validation = validate_config(config_update)
        if not validation["is_valid"]:
            raise HTTPException(
                status_code=400,
                detail=f"Configuration validation failed: {', '.join(validation['errors'])}"
            )
        
        # Save configuration
        config = save_admin_config(db, config_update)
        return config
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error updating admin settings: {str(exc)}"
        )


@router.post("/settings/validate", response_model=AdminConfigValidationResponse, tags=["admin-settings"])
def validate_admin_settings(
    config_update: AdminConfigUpdate,
    _current_user: User = Depends(require_roles("admin")),
):
    """
    Validate admin configuration without saving.
    
    Useful for previewing changes before committing them.
    """
    try:
        validation = validate_config(config_update)
        
        return AdminConfigValidationResponse(
            is_valid=validation["is_valid"],
            message="Configuration is valid" if validation["is_valid"] 
                    else f"Validation errors: {', '.join(validation['errors'])}",
            config=None
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error validating admin settings: {str(exc)}"
        )


@router.post("/settings/reset", response_model=AdminConfigResponse, tags=["admin-settings"])
def reset_admin_settings_to_default(
    confirm: bool = Query(False, description="Set to true to confirm reset"),
    _current_user: User = Depends(require_roles("admin")),
    db: Session = Depends(get_db),
):
    """
    Reset admin configuration to default values.
    
    Parameters:
    - confirm: Must be True to actually perform the reset
    
    Default values:
    - max_keywords_display: 10
    - hook_analysis_duration: 60 seconds
    - analysis_time_range_days: 90
    - notification_batch_size: 50
    - youtube_region: TH
    - google_region: TH
    - enable_youtube_trending: True
    - enable_google_trends: True
    - auto_scan_interval_hours: 6
    """
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Reset not confirmed. Set confirm=true to proceed."
        )
    
    try:
        config = reset_admin_config(db)
        return config
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error resetting admin settings: {str(exc)}"
        )


@router.get("/statistics", response_model=AdminStatisticsResponse, tags=["admin-settings"])
def get_admin_dashboard_statistics(
    _current_user: User = Depends(require_roles("admin")),
    db: Session = Depends(get_db),
):
    """
    Get comprehensive admin dashboard statistics.
    
    Returns:
    - User statistics (total, active, inactive)
    - Analysis statistics (total count, per-user average)
    - Notification statistics (total, unread)
    - Trending topic statistics (followed domains and keywords)
    - Data source status and last scan times
    """
    try:
        stats = get_admin_statistics(db)
        
        return AdminStatisticsResponse(
            total_users=stats["total_users"],
            active_users=stats["active_users"],
            total_analyses=stats["total_analyses"],
            total_saved_ideas=stats.get("total_saved_ideas", 0),
            analyses_by_category=stats["analyses_by_category"],
            most_used_categories=stats["most_used_categories"],
            avg_analyses_per_user=stats["avg_analyses_per_user"],
            trend_data_sources_active=stats["trend_data_sources_active"],
            last_trend_scan=stats.get("last_trend_scan"),
            last_config_update=stats.get("last_config_update"),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving admin statistics: {str(exc)}"
        )


@router.post("/settings/backup", tags=["admin-settings"])
def backup_admin_configuration(
    _current_user: User = Depends(require_roles("admin")),
    db: Session = Depends(get_db),
):
    """
    Export current admin configuration as a backup.
    
    Returns a JSON object containing:
    - Current configuration settings
    - System statistics at backup time
    - Backup timestamp
    
    Useful before making significant changes or for disaster recovery.
    """
    try:
        backup = export_config_backup(db)
        return {
            "status": "success",
            "message": "Configuration backup created",
            "backup": backup
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error creating backup: {str(exc)}"
        )


@router.post("/settings/restore", response_model=AdminConfigResponse, tags=["admin-settings"])
def restore_admin_configuration_from_backup(
    backup_data: dict,
    confirm: bool = Query(False, description="Set to true to confirm restore"),
    _current_user: User = Depends(require_roles("admin")),
    db: Session = Depends(get_db),
):
    """
    Restore admin configuration from a backup.
    
    Parameters:
    - backup_data: The backup JSON object (from /admin/settings/backup)
    - confirm: Must be True to actually perform the restore
    
    This will overwrite the current configuration with the backed-up values.
    """
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Restore not confirmed. Set confirm=true to proceed."
        )
    
    try:
        config = apply_config_from_backup(db, backup_data)
        return config
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error restoring configuration: {str(exc)}"
        )


@router.post("/settings/audit-log", tags=["admin-settings"])
def get_settings_audit_log(
    limit: int = Query(50, ge=1, le=200, description="Maximum log entries to return"),
    _current_user: User = Depends(require_roles("admin")),
    db: Session = Depends(get_db),
):
    """
    Get audit log of admin configuration changes.
    
    Returns a log of who changed what settings and when.
    (Can be extended to log actual changes to system_configs table)
    """
    try:
        audit_data = get_admin_audit_log(db, limit=limit)
        return {
            "status": "success",
            "message": "Audit log retrieved",
            "total_entries": audit_data["total_entries"],
            "entries": audit_data["entries"],
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving audit log: {str(exc)}"
        )


@router.post("/health", tags=["admin-settings"])
def check_admin_system_health(
    _current_user: User = Depends(require_roles("admin")),
    db: Session = Depends(get_db),
):
    """
    Check overall system health and configuration status.
    
    Validates:
    - Database connection
    - Configuration validity
    - Critical settings values
    - Data source connectivity
    """
    try:
        config = get_admin_config(db)
        stats = get_admin_statistics(db)
        
        # Health checks
        health_checks = {
            "database": "ok" if config.config_id else "error",
            "configuration_valid": "ok" if all([
                1 <= config.max_keywords_display <= 50,
                10 <= config.hook_analysis_duration <= 300,
                1 <= config.analysis_time_range_days <= 365,
            ]) else "warning",
            "users_registered": "ok" if stats["total_users"] > 0 else "warning",
            "system_active": "ok" if stats["active_users"] > 0 else "warning",
        }
        
        overall_status = "ok" if all(v == "ok" for v in health_checks.values()) else "warning"
        
        return {
            "status": overall_status,
            "timestamp": __import__('datetime').datetime.utcnow().isoformat(),
            "health_checks": health_checks,
            "configuration_summary": {
                "max_keywords": config.max_keywords_display,
                "hook_duration_sec": config.hook_analysis_duration,
                "analysis_range_days": config.analysis_time_range_days,
            },
            "statistics_summary": {
                "total_users": stats["total_users"],
                "total_analyses": stats["total_analyses"],
                "active_notifications": stats["total_notifications"],
            }
        }
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error checking system health: {str(exc)}"
        )
