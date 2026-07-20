from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class DatasetContentItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    dataset_id: int
    title: str
    video_url: Optional[str] = None
    transcript: Optional[str] = None
    category: Optional[str] = None
    source_platform: str
    views: int
    likes: int
    comments: int
    trend_score: float
    duration_seconds: Optional[int] = None
    published_at: Optional[datetime] = None
    created_at: datetime


class DatasetListResponse(BaseModel):
    source: str
    total: int
    limit: int
    offset: int
    items: list[DatasetContentItem]
