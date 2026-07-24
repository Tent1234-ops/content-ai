from datetime import datetime
from typing import List

from pydantic import BaseModel, ConfigDict


class NotificationItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    notification_id: int
    user_id: int
    title: str
    body: str | None = None
    link: str | None = None
    type: str
    is_read: bool
    delivered_via_ws: bool
    created_at: datetime


class NotificationsResponse(BaseModel):
    total: int
    items: List[NotificationItem]


class MarkReadRequest(BaseModel):
    ids: list[int]
