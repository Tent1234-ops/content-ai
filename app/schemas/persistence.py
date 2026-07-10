from pydantic import BaseModel


class TrendSyncResponse(BaseModel):
    created: int
    updated: int
    mode: str
    region: str
    total_fetched: int
