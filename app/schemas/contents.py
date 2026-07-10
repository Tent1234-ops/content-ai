from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel


class UserContentHistoryItem(BaseModel):
    content_id: int
    title: str
    created_at: datetime
    video_url: Optional[str] = None
    transcript_preview: Optional[str] = None
    domain: Optional[str] = None
    recommended_duration: Optional[int] = None
    recommended_keywords: list[str]
    hook_keywords: list[str]


class UserContentHistoryResponse(BaseModel):
    total: int
    items: list[UserContentHistoryItem]


class UserContentDetailResponse(BaseModel):
    content_id: int
    title: str
    created_at: datetime
    video_url: Optional[str] = None
    transcript: Optional[str] = None
    analysis: dict[str, Any]
    nlp_result: dict[str, Any]
    recommendation: dict[str, Any]
