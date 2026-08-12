from __future__ import annotations


YOUTUBE_CC_DATASET_SOURCE = "youtube_cc"
DEFAULT_YOUTUBE_CC_DATASET_VERSION = "youtube-cc-th-v1"
YOUTUBE_CC_LABEL_SOURCE = "human_review"
YOUTUBE_CC_VERIFICATION_STATUS = "human_verified"
YOUTUBE_CC_LICENSE_NAME = "YouTube Creative Commons Attribution"
YOUTUBE_CC_LICENSE_URL = "https://support.google.com/youtube/answer/2797468"
YOUTUBE_CC_TRANSCRIPT_SOURCE = "youtube_public_caption"

SUPPORTED_TRANSCRIPT_LANGUAGES = ("th", "en")
SUPPORTED_CAPTION_TYPES = ("manual", "auto_generated")
ACCEPTED_TRANSCRIPT_QUALITIES = ("good", "acceptable")
PRODUCTION_SPLITS = ("train", "validation", "test")
TRANSCRIPT_WINDOW_SECONDS = 300
MAX_VIDEO_DURATION_SECONDS = 300
SPLIT_STRATEGY = "channel_sha256_bucket_v1_70_15_15"

