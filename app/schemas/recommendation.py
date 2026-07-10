from typing import Optional

from pydantic import BaseModel, Field

from app.schemas.classification import ClassificationResponse


class RecommendationKeywordItem(BaseModel):
    keyword: str
    score: float


class RecommendationDimensionItem(BaseModel):
    name: str
    score: float
    user_status: str


class DurationRecommendation(BaseModel):
    recommended_seconds: int
    recommended_range: str
    sample_size: int
    source: str


class DatasetProfileResponseItem(BaseModel):
    domain: str
    sample_size: int
    top_keywords: list[RecommendationKeywordItem]
    top_dimensions: list[RecommendationKeywordItem]
    hook_keywords: list[RecommendationKeywordItem]
    recommended_duration: DurationRecommendation
    exemplar_titles: list[str]


class DatasetProfilesResponse(BaseModel):
    source: str
    total_profiles: int
    profiles: list[DatasetProfileResponseItem]


class RecommendationTextRequest(BaseModel):
    title: Optional[str] = Field(default=None, max_length=255)
    text: str = Field(..., min_length=1)
    source: str = Field(default="youtube", pattern="^(youtube|google)$")
    profile_limit: int = Field(default=100, ge=10, le=500)


class RecommendationAnalysisResponse(BaseModel):
    domain: str
    classification: Optional[ClassificationResponse] = None
    user_keywords: list[str]
    missing_keywords: list[RecommendationKeywordItem]
    hook_keywords: list[RecommendationKeywordItem]
    missing_dimensions: list[RecommendationDimensionItem]
    recommended_duration: DurationRecommendation
    dataset_profile: DatasetProfileResponseItem


class ProfileComparisonItem(BaseModel):
    domain: str
    left_sample_size: int
    right_sample_size: int
    left_top_keywords: list[RecommendationKeywordItem]
    right_top_keywords: list[RecommendationKeywordItem]
    left_duration: DurationRecommendation
    right_duration: DurationRecommendation


class ProfileComparisonResponse(BaseModel):
    left_source: str
    right_source: str
    comparisons: list[ProfileComparisonItem]


class SourceActivityItem(BaseModel):
    source_platform: str
    count: int


class RecommendationAdminDatasetHealth(BaseModel):
    total_dataset_contents: int
    youtube_dataset_contents: int
    google_dataset_contents: int
    duration_coverage_count: int
    duration_coverage_ratio: float


class RecommendationAdminProfileHealth(BaseModel):
    youtube_profiles: int
    google_profiles: int
    youtube_domains: list[str]
    google_domains: list[str]


class RecommendationAdminReportResponse(BaseModel):
    dataset_health: RecommendationAdminDatasetHealth
    profile_health: RecommendationAdminProfileHealth
    recent_source_activity: list[SourceActivityItem]
    youtube_profiles: list[DatasetProfileResponseItem]
    google_profiles: list[DatasetProfileResponseItem]
