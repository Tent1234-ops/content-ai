from typing import Optional

from pydantic import BaseModel, Field


class ClassificationRequest(BaseModel):
    title: Optional[str] = Field(default=None, max_length=255)
    text: str = Field(..., min_length=1)
    source: str = Field(default="youtube", pattern="^(youtube|google)$")
    profile_limit: int = Field(default=200, ge=10, le=500)
    top_k: int = Field(default=5, ge=1, le=10)


class ClassificationCandidate(BaseModel):
    domain: str
    score: float
    similarity: float
    sample_size: int
    matched_terms: list[str]


class ClassificationResponse(BaseModel):
    domain: str
    confidence: float
    method: str
    rule_domain: str
    source: str
    profile_limit: int
    candidates: list[ClassificationCandidate]
