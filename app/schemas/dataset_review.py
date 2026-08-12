from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class DatasetReviewRunItem(BaseModel):
    collection_run_id: int
    dataset_version: str
    status: str
    started_at: datetime
    total: int
    pending: int
    approved: int
    rejected: int


class DatasetReviewCandidateItem(BaseModel):
    collection_run_id: int
    dataset_version: str
    source_youtube_id: str
    candidate_sha256: str
    title: str
    video_url: str
    channel_title: str
    proposed_leaf_key: str
    transcript_language: str
    caption_type: str
    duration_seconds: int
    transcript: str
    transcript_preview: str
    evidence_terms: list[str]
    automated_checks: dict[str, bool]
    views: int
    likes: int
    comments: int
    review_status: Literal["pending", "approved", "rejected"]
    reviewed_leaf_key: str | None = None
    transcript_quality: str | None = None
    reviewer: str | None = None
    reviewed_at: datetime | None = None
    review_notes: str | None = None
    dataset_id: int | None = None


class DatasetReviewQueueResponse(BaseModel):
    total: int
    limit: int
    offset: int
    summary: dict[str, int]
    runs: list[DatasetReviewRunItem]
    taxonomy: list[dict[str, Any]]
    items: list[DatasetReviewCandidateItem]


class DatasetReviewDecisionRequest(BaseModel):
    decision: Literal["approve", "reject"]
    reviewed_leaf_key: str | None = Field(default=None, max_length=100)
    transcript_quality: Literal["good", "acceptable"] | None = None
    notes: str | None = Field(default=None, max_length=2000)


class DatasetReviewDecisionResponse(BaseModel):
    status: str
    decision: Literal["approve", "reject"]
    source_youtube_id: str
    collection_run_id: int
    dataset_id: int | None = None
    review_event_id: int | None = None
    run_status: str
    coverage: dict[str, Any]
