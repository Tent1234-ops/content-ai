from __future__ import annotations

import hashlib


YOUTUBE_CC_DATASET_SOURCE = "youtube_cc"
DEFAULT_YOUTUBE_CC_DATASET_VERSION = "youtube-cc-th-v1"
YOUTUBE_CC_LABEL_SOURCE = "human_review"
YOUTUBE_CC_VERIFICATION_STATUS = "human_verified"
YOUTUBE_CC_LICENSE_NAME = "YouTube Creative Commons Attribution"
YOUTUBE_CC_LICENSE_URL = "https://support.google.com/youtube/answer/2797468"
YOUTUBE_CC_TRANSCRIPT_SOURCE = "youtube_public_caption"
YOUTUBE_TRANSCRIPT_API_ACQUISITION = "youtube_transcript_api"
NOTEBOOKLM_TRANSCRIPT_ACQUISITION = "notebooklm_manual_source"
SUPPORTED_TRANSCRIPT_ACQUISITION_METHODS = (
    YOUTUBE_TRANSCRIPT_API_ACQUISITION,
    NOTEBOOKLM_TRANSCRIPT_ACQUISITION,
)
TRANSCRIPT_SCOPE_FIRST_WINDOW = "first_window"
TRANSCRIPT_SCOPE_FULL_VIDEO = "full_video"
SUPPORTED_TRANSCRIPT_SCOPES = (
    TRANSCRIPT_SCOPE_FIRST_WINDOW,
    TRANSCRIPT_SCOPE_FULL_VIDEO,
)

SUPPORTED_TRANSCRIPT_LANGUAGES = ("th", "en")
PRIMARY_CONTENT_LANGUAGE = "th"
DEFAULT_COLLECTION_LANGUAGES = (PRIMARY_CONTENT_LANGUAGE,)
SUPPORTED_CAPTION_TYPES = ("manual", "auto_generated", "unspecified")
ACCEPTED_TRANSCRIPT_QUALITIES = ("good", "acceptable")
PRODUCTION_SPLITS = ("train", "validation", "test")
TRANSCRIPT_WINDOW_SECONDS = 300
USER_UPLOAD_MAX_DURATION_SECONDS = 300
RECOMMENDATION_DURATION_MAX_SECONDS = 300
SPLIT_STRATEGY = "channel_sha256_bucket_v1_70_15_15"


def channel_dataset_split(channel_id: str) -> tuple[str, str]:
    """Return a stable split and anonymized group key for one YouTube channel."""
    normalized_channel_id = str(channel_id or "").strip()
    if not normalized_channel_id:
        raise ValueError("channel_id is required for channel-grouped dataset splits")
    creator_group_key = hashlib.sha256(normalized_channel_id.encode("utf-8")).hexdigest()
    bucket = int(creator_group_key[:8], 16) % 100
    if bucket < 70:
        split = "train"
    elif bucket < 85:
        split = "validation"
    else:
        split = "test"
    return split, creator_group_key
