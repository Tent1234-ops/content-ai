from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class AdminDatasetItem(BaseModel):
    dataset_id: int
    title: str
    category: Optional[str] = None
    source_platform: str
    trend_score: float
    views: int
    likes: int
    comments: int
    duration_seconds: Optional[int] = None
    published_at: Optional[datetime] = None
    created_at: datetime


class AdminDatasetListResponse(BaseModel):
    total: int
    items: list[AdminDatasetItem]


class AdminClusterRunItem(BaseModel):
    run_id: int
    algorithm: str
    n_clusters: int
    feature_dimension: int
    inertia: float
    membership_count: int
    created_at: datetime


class AdminClusterRunListResponse(BaseModel):
    total: int
    items: list[AdminClusterRunItem]


class AdminSystemLogItem(BaseModel):
    log_id: int
    action: str
    status: str
    detail: Optional[str] = None
    timestamp: datetime
    user_id: Optional[int] = None


class AdminSystemLogListResponse(BaseModel):
    total: int
    items: list[AdminSystemLogItem]


class AdminSourceTrendItem(BaseModel):
    source_platform: str
    count: int
    avg_trend_score: float


class AdminStatusBreakdownItem(BaseModel):
    status: str
    count: int


class AdminClusterBreakdownItem(BaseModel):
    cluster_id: int
    cluster_name: str
    member_count: int


class AdminClusterMembershipItem(BaseModel):
    membership_id: int
    cluster_id: int
    cluster_name: str
    dataset_id: Optional[int] = None
    content_id: Optional[int] = None
    item_text: str
    top_terms: Optional[str] = None
    created_at: datetime


class AdminClusterRunDetailResponse(BaseModel):
    run_id: int
    algorithm: str
    n_clusters: int
    feature_dimension: int
    inertia: float
    membership_count: int
    created_at: datetime
    cluster_breakdown: list[AdminClusterBreakdownItem]
    recent_memberships: list[AdminClusterMembershipItem]


class AdminOverviewReportResponse(BaseModel):
    dataset_total: int
    cluster_run_total: int
    system_log_total: int
    top_sources: list[AdminSourceTrendItem]
    top_categories: list[dict]
    status_breakdown: list[AdminStatusBreakdownItem]
    recent_actions: list[AdminSystemLogItem]
