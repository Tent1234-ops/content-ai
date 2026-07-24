from datetime import datetime
from typing import List

from pydantic import BaseModel, ConfigDict, Field


class FollowTopicRequest(BaseModel):
    match_type: str = Field(..., description="'domain' or 'keyword'")
    value: str


class FollowedTopicItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    match_type: str
    value: str
    created_at: datetime


class FollowedTopicsResponse(BaseModel):
    total: int
    items: List[FollowedTopicItem]
