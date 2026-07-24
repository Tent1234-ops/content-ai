"""Admin configuration service for managing system-wide settings"""

from typing import Dict, Optional, List
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.database.models import SystemConfig, User, UserContent, FollowedTopic, Notification
from app.schemas.admin_config import AdminConfigUpdate, AdminConfigResponse


# Default configuration values
DEFAULT_CONFIG = {
    "max_keywords_display": 10,
    "hook_analysis_duration": 60,
    "analysis_time_range_days": 90,
    "notification_batch_size": 50,
    "youtube_region": "TH",
    "google_region": "TH",
    "enable_youtube_trending": True,
    "enable_google_trends": True,
    "auto_scan_interval_hours": 6,
}


def get_or_create_admin_config(db: Session) -> SystemConfig:
    """
    Get admin configuration or create default if not exists.
    Admin config is shared globally (single config_id = None for system-wide)
    """
    config = db.query(SystemConfig).filter(SystemConfig.user_id == None).first()
    
    if config:
        return config
    
    # Create default config
    config = SystemConfig(
        user_id=None,  # System-wide config
        max_keywords=DEFAULT_CONFIG["max_keywords_display"],
        hook_duration=DEFAULT_CONFIG["hook_analysis_duration"],
        process_interval=DEFAULT_CONFIG["analysis_time_range_days"],
    )
    db.add(config)
    db.commit()
    db.refresh(config)
    return config


def get_admin_config(db: Session) -> AdminConfigResponse:
    """
    Retrieve current admin configuration
    """
    config = get_or_create_admin_config(db)
    
    # Map database fields to response schema
    return AdminConfigResponse(
        config_id=config.config_id,
        max_keywords_display=config.max_keywords,
        hook_analysis_duration=config.hook_duration,
        analysis_time_range_days=config.process_interval,
        notification_batch_size=50,  # Default, can be extended to DB
        youtube_region="TH",  # Can be extended to DB
        google_region="TH",  # Can be extended to DB
        enable_youtube_trending=True,  # Can be extended to DB
        enable_google_trends=True,  # Can be extended to DB
        auto_scan_interval_hours=6,  # Can be extended to DB
        created_at=config.created_at,
        updated_at=config.created_at,
    )


def save_admin_config(db: Session, config_update: AdminConfigUpdate) -> AdminConfigResponse:
    """
    Save/update admin configuration.
    Only updates fields that are provided (not None).
    """
    config = get_or_create_admin_config(db)
    
    # Update only provided fields
    update_data = config_update.model_dump(exclude_unset=True)
    
    for field, value in update_data.items():
        if value is not None:
            # Map schema field names to database field names
            if field == "max_keywords_display":
                config.max_keywords = value
            elif field == "hook_analysis_duration":
                config.hook_duration = value
            elif field == "analysis_time_range_days":
                config.process_interval = value
            # Other fields (notification_batch_size, regions, etc.) can be extended to DB
    
    db.commit()
    db.refresh(config)
    
    return AdminConfigResponse(
        config_id=config.config_id,
        max_keywords_display=config.max_keywords,
        hook_analysis_duration=config.hook_duration,
        analysis_time_range_days=config.process_interval,
        notification_batch_size=50,
        youtube_region="TH",
        google_region="TH",
        enable_youtube_trending=True,
        enable_google_trends=True,
        auto_scan_interval_hours=6,
        created_at=config.created_at,
        updated_at=datetime.utcnow(),
    )


def reset_admin_config(db: Session) -> AdminConfigResponse:
    """
    Reset admin configuration to defaults
    """
    config = get_or_create_admin_config(db)
    
    config.max_keywords = DEFAULT_CONFIG["max_keywords_display"]
    config.hook_duration = DEFAULT_CONFIG["hook_analysis_duration"]
    config.process_interval = DEFAULT_CONFIG["analysis_time_range_days"]
    
    db.commit()
    db.refresh(config)
    
    return AdminConfigResponse(
        config_id=config.config_id,
        max_keywords_display=config.max_keywords,
        hook_analysis_duration=config.hook_duration,
        analysis_time_range_days=config.process_interval,
        notification_batch_size=50,
        youtube_region="TH",
        google_region="TH",
        enable_youtube_trending=True,
        enable_google_trends=True,
        auto_scan_interval_hours=6,
        created_at=config.created_at,
        updated_at=datetime.utcnow(),
    )


def validate_config(config_update: AdminConfigUpdate) -> Dict[str, bool]:
    """
    Validate configuration values
    Returns dict with validation result and any error messages
    """
    errors = []
    
    if config_update.max_keywords_display is not None:
        if not (1 <= config_update.max_keywords_display <= 50):
            errors.append("max_keywords_display must be between 1 and 50")
    
    if config_update.hook_analysis_duration is not None:
        if not (10 <= config_update.hook_analysis_duration <= 300):
            errors.append("hook_analysis_duration must be between 10 and 300 seconds")
    
    if config_update.analysis_time_range_days is not None:
        if not (1 <= config_update.analysis_time_range_days <= 365):
            errors.append("analysis_time_range_days must be between 1 and 365")
    
    if config_update.notification_batch_size is not None:
        if not (1 <= config_update.notification_batch_size <= 200):
            errors.append("notification_batch_size must be between 1 and 200")
    
    if config_update.auto_scan_interval_hours is not None:
        if not (1 <= config_update.auto_scan_interval_hours <= 24):
            errors.append("auto_scan_interval_hours must be between 1 and 24")
    
    return {
        "is_valid": len(errors) == 0,
        "errors": errors,
    }


def get_admin_statistics(db: Session) -> Dict:
    """
    Calculate admin dashboard statistics
    """
    total_users = db.query(User).count()
    active_users = db.query(User).filter(User.is_active == True).count()
    total_analyses = db.query(UserContent).count()
    
    # Count analyses by category (from recommendations or analysis results)
    # This is a simplified version - extend based on actual schema
    categories_result = db.query(
        func.count(UserContent.content_id)
    ).first()
    
    # Get most used categories (top 5)
    most_used = []
    
    # Get notification stats
    total_notifications = db.query(Notification).count()
    unread_notifications = db.query(Notification).filter(Notification.is_read == False).count()
    
    # Get followed topics
    followed_domains = db.query(FollowedTopic).filter(
        FollowedTopic.match_type == "domain"
    ).count()
    followed_keywords = db.query(FollowedTopic).filter(
        FollowedTopic.match_type == "keyword"
    ).count()
    
    avg_analyses = round(total_analyses / max(total_users, 1), 2)
    
    return {
        "total_users": total_users,
        "active_users": active_users,
        "inactive_users": total_users - active_users,
        "total_analyses": total_analyses,
        "avg_analyses_per_user": avg_analyses,
        "total_notifications": total_notifications,
        "unread_notifications": unread_notifications,
        "followed_domains": followed_domains,
        "followed_keywords": followed_keywords,
        "analyses_by_category": {},
        "most_used_categories": most_used,
        "trend_data_sources_active": ["youtube", "google"],
        "system_uptime_hours": 24,  # Placeholder
    }


def export_config_backup(db: Session) -> Dict:
    """
    Export current configuration as backup
    """
    config = get_or_create_admin_config(db)
    stats = get_admin_statistics(db)
    
    return {
        "backup_timestamp": datetime.utcnow().isoformat(),
        "configuration": {
            "config_id": config.config_id,
            "max_keywords": config.max_keywords,
            "hook_duration": config.hook_duration,
            "process_interval": config.process_interval,
        },
        "statistics": stats,
    }


def apply_config_from_backup(db: Session, backup_config: Dict) -> AdminConfigResponse:
    """
    Apply configuration from backup data
    """
    config = get_or_create_admin_config(db)
    
    if "configuration" in backup_config:
        cfg = backup_config["configuration"]
        if "max_keywords" in cfg:
            config.max_keywords = cfg["max_keywords"]
        if "hook_duration" in cfg:
            config.hook_duration = cfg["hook_duration"]
        if "process_interval" in cfg:
            config.process_interval = cfg["process_interval"]
    
    db.commit()
    db.refresh(config)
    
    return AdminConfigResponse(
        config_id=config.config_id,
        max_keywords_display=config.max_keywords,
        hook_analysis_duration=config.hook_duration,
        analysis_time_range_days=config.process_interval,
        notification_batch_size=50,
        youtube_region="TH",
        google_region="TH",
        enable_youtube_trending=True,
        enable_google_trends=True,
        auto_scan_interval_hours=6,
        created_at=config.created_at,
        updated_at=datetime.utcnow(),
    )
