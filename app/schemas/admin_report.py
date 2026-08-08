from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class AdminDatasetItem(BaseModel):
    dataset_id: int
    title: str
    video_url: Optional[str] = None
    transcript: Optional[str] = None
    category: Optional[str] = None
    source_platform: str
    trend_score: float
    views: int
    likes: int
    comments: int
    duration_seconds: Optional[int] = None
    published_at: Optional[datetime] = None
    created_at: datetime


class AdminDatasetCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=255)
    video_url: Optional[str] = None
    transcript: Optional[str] = None
    category: Optional[str] = None
    source_platform: str = Field(default="youtube_admin", min_length=1, max_length=50)
    views: int = Field(default=0, ge=0)
    likes: int = Field(default=0, ge=0)
    comments: int = Field(default=0, ge=0)
    trend_score: float = Field(default=0.0, ge=0)
    duration_seconds: Optional[int] = Field(default=None, ge=0)
    published_at: Optional[datetime] = None


class AdminDatasetUpdate(BaseModel):
    title: Optional[str] = Field(default=None, min_length=1, max_length=255)
    video_url: Optional[str] = None
    transcript: Optional[str] = None
    category: Optional[str] = None
    source_platform: Optional[str] = Field(default=None, min_length=1, max_length=50)
    views: Optional[int] = Field(default=None, ge=0)
    likes: Optional[int] = Field(default=None, ge=0)
    comments: Optional[int] = Field(default=None, ge=0)
    trend_score: Optional[float] = Field(default=None, ge=0)
    duration_seconds: Optional[int] = Field(default=None, ge=0)
    published_at: Optional[datetime] = None


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
