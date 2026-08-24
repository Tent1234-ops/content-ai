from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import relationship

from .db import Base


class User(Base):
    __tablename__ = "users"

    user_id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(20), nullable=False, default="user")
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    contents = relationship("UserContent", back_populates="owner", cascade="all, delete-orphan")
    logs = relationship("SystemLog", back_populates="user")
    configs = relationship("SystemConfig", back_populates="user")
    notifications = relationship("Notification", back_populates="user", cascade="all, delete-orphan")
    followed_topics = relationship("FollowedTopic", back_populates="user", cascade="all, delete-orphan")
    trend_watch_sessions = relationship(
        "UserTrendWatchSession",
        back_populates="user",
        cascade="all, delete-orphan",
    )


class UserContent(Base):
    __tablename__ = "user_contents"

    content_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False)
    title = Column(String(255), nullable=False)
    video_url = Column(String(1024))
    transcript = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    owner = relationship("User", back_populates="contents")
    analysis_results = relationship("AnalysisResult", back_populates="content", cascade="all, delete-orphan")
    recommendations = relationship("Recommendation", back_populates="content", cascade="all, delete-orphan")
    content_keywords = relationship("ContentKeyword", back_populates="content", cascade="all, delete-orphan")


class Keyword(Base):
    __tablename__ = "keywords"

    keyword_id = Column(Integer, primary_key=True)
    keyword = Column(String(100), unique=True, nullable=False)

    content_keywords = relationship("ContentKeyword", back_populates="keyword_ref")


class ContentKeyword(Base):
    __tablename__ = "content_keywords"

    id = Column(Integer, primary_key=True)
    content_id = Column(Integer, ForeignKey("user_contents.content_id"), nullable=False)
    keyword_id = Column(Integer, ForeignKey("keywords.keyword_id"), nullable=False)
    score = Column(Float, nullable=False, default=0.0)

    content = relationship("UserContent", back_populates="content_keywords")
    keyword_ref = relationship("Keyword", back_populates="content_keywords")


class Cluster(Base):
    __tablename__ = "clusters"

    cluster_id = Column(Integer, primary_key=True)
    cluster_name = Column(String(150), nullable=False, unique=True)
    description = Column(Text)

    analysis_results = relationship("AnalysisResult", back_populates="cluster")
    memberships = relationship("ClusterMembership", back_populates="cluster", cascade="all, delete-orphan")


class TaxonomyNode(Base):
    __tablename__ = "taxonomy_nodes"
    __table_args__ = (
        UniqueConstraint("taxonomy_version", "node_key", name="uq_taxonomy_version_node_key"),
        Index("ix_taxonomy_nodes_version_level", "taxonomy_version", "level"),
    )

    taxonomy_node_id = Column(Integer, primary_key=True)
    taxonomy_version = Column(String(50), nullable=False, index=True)
    node_key = Column(String(100), nullable=False)
    display_name = Column(String(150), nullable=False)
    display_name_th = Column(String(150))
    level = Column(Integer, nullable=False)
    parent_key = Column(String(100))
    is_leaf = Column(Boolean, nullable=False, default=False)
    is_active = Column(Boolean, nullable=False, default=True)
    is_trainable = Column(Boolean, nullable=False, default=False)
    minimum_sample_count = Column(Integer, nullable=False, default=0)
    source_dataset = Column(String(100))
    source_category = Column(String(100))
    source_subcategory = Column(Text)
    mapping_rule = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


class DatasetCollectionRun(Base):
    __tablename__ = "dataset_collection_runs"

    collection_run_id = Column(Integer, primary_key=True)
    run_key = Column(String(64), nullable=False, unique=True, index=True)
    dataset_source = Column(
        String(100), nullable=False, default="youtube_public_research"
    )
    dataset_version = Column(String(100), nullable=False)
    status = Column(String(30), nullable=False, default="running")
    region_code = Column(String(10), nullable=False, default="TH")
    languages_json = Column(Text, nullable=False)
    query_config_json = Column(Text, nullable=False)
    candidate_artifact_path = Column(String(1024))
    candidate_artifact_sha256 = Column(String(64))
    review_artifact_path = Column(String(1024))
    review_artifact_sha256 = Column(String(64))
    manifest_path = Column(String(1024))
    manifest_sha256 = Column(String(64))
    candidates_seen = Column(Integer, nullable=False, default=0)
    transcripts_collected = Column(Integer, nullable=False, default=0)
    duplicates_skipped = Column(Integer, nullable=False, default=0)
    errors_count = Column(Integer, nullable=False, default=0)
    resume_count = Column(Integer, nullable=False, default=0)
    last_resumed_at = Column(DateTime)
    started_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    contents = relationship("DatasetContent", back_populates="collection_run")
    review_events = relationship(
        "DatasetReviewEvent",
        back_populates="collection_run",
        cascade="all, delete-orphan",
    )


class DatasetContent(Base):
    __tablename__ = "dataset_contents"
    __table_args__ = (
        UniqueConstraint(
            "dataset_source",
            "dataset_version",
            "source_record_id",
            name="uq_dataset_source_version_record",
        ),
        UniqueConstraint("source_youtube_id", name="uq_dataset_source_youtube_id"),
        UniqueConstraint("transcript_sha256", name="uq_dataset_transcript_sha256"),
        Index("ix_dataset_taxonomy_leaf", "taxonomy_version", "taxonomy_leaf_key"),
        Index(
            "ix_dataset_training_eligibility",
            "is_training_eligible",
            "data_split",
            "taxonomy_leaf_key",
        ),
        Index(
            "ix_dataset_view_metric_leaf",
            "view_metric_version",
            "taxonomy_leaf_key",
        ),
    )

    dataset_id = Column(Integer, primary_key=True)
    title = Column(String(255), nullable=False)
    video_url = Column(String(1024))
    transcript = Column(Text)
    category = Column(String(100))
    source_platform = Column(String(50), nullable=False, default="youtube")
    dataset_source = Column(String(100), nullable=False, default="legacy")
    dataset_version = Column(String(100), nullable=False, default="legacy-v1")
    collection_run_id = Column(
        Integer,
        ForeignKey("dataset_collection_runs.collection_run_id", ondelete="SET NULL"),
        index=True,
    )
    source_record_id = Column(String(255))
    source_youtube_id = Column(String(32))
    source_creator = Column(String(255))
    source_channel_id = Column(String(64), index=True)
    source_category = Column(String(100))
    source_subcategory = Column(String(100))
    collection_query = Column(String(255))
    source_release_url = Column(String(1024))
    source_archive_sha256 = Column(String(64))
    source_annotation_path = Column(String(1024))
    source_annotation_sha256 = Column(String(64))
    import_batch_id = Column(String(64))
    taxonomy_version = Column(String(50), nullable=False, default="legacy-v1")
    taxonomy_leaf_key = Column(String(100), index=True)
    category_level_1 = Column(String(150))
    category_level_2 = Column(String(150))
    category_level_3 = Column(String(150))
    language = Column(String(20), nullable=False, default="und")
    verification_status = Column(String(30), nullable=False, default="unverified")
    label_source = Column(String(100), nullable=False, default="unverified")
    license_name = Column(String(100), nullable=False, default="unknown")
    license_url = Column(String(1024))
    data_split = Column(String(20), nullable=False, default="unassigned")
    split_strategy = Column(String(100))
    creator_group_key = Column(String(64), index=True)
    transcript_sha256 = Column(String(64))
    transcript_segment_count = Column(Integer, nullable=False, default=0)
    transcript_start_seconds = Column(Float)
    transcript_end_seconds = Column(Float)
    transcript_window_seconds = Column(Integer)
    transcript_source = Column(String(50))
    transcript_acquisition_method = Column(
        String(64), nullable=False, default="youtube_transcript_api"
    )
    transcript_scope = Column(String(32), nullable=False, default="first_window")
    transcript_timestamps_available = Column(Boolean, nullable=False, default=True)
    caption_type = Column(String(50))
    transcript_quality = Column(String(30))
    reviewed_by = Column(String(255))
    reviewed_at = Column(DateTime)
    review_notes = Column(Text)
    statistics_captured_at = Column(DateTime)
    view_metric_version = Column(
        String(64),
        nullable=False,
        default="unknown_v1",
    )
    license_verified_at = Column(DateTime)
    raw_metadata_json = Column(Text)
    collection_strategy = Column(String(50))
    average_views_per_day = Column(Float, nullable=False, default=0.0)
    engagement_rate = Column(Float, nullable=False, default=0.0)
    is_training_eligible = Column(Boolean, nullable=False, default=False)
    is_keyword_recommendation_eligible = Column(
        Boolean,
        nullable=False,
        default=False,
    )
    is_duration_recommendation_eligible = Column(
        Boolean,
        nullable=False,
        default=False,
    )
    is_active = Column(Boolean, nullable=False, default=True)
    views = Column(Integer, nullable=False, default=0)
    likes = Column(Integer, nullable=False, default=0)
    comments = Column(Integer, nullable=False, default=0)
    trend_score = Column(Float, nullable=False, default=0.0)
    duration_seconds = Column(Integer)
    published_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    analysis_results = relationship("AnalysisResult", back_populates="dataset")
    memberships = relationship("ClusterMembership", back_populates="dataset")
    collection_run = relationship("DatasetCollectionRun", back_populates="contents")
    review_events = relationship("DatasetReviewEvent", back_populates="dataset")


class DatasetReviewEvent(Base):
    __tablename__ = "dataset_review_events"
    __table_args__ = (
        Index("ix_dataset_review_source_video", "source_youtube_id", "reviewed_at"),
    )

    review_event_id = Column(Integer, primary_key=True)
    collection_run_id = Column(
        Integer,
        ForeignKey("dataset_collection_runs.collection_run_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    dataset_id = Column(
        Integer,
        ForeignKey("dataset_contents.dataset_id", ondelete="SET NULL"),
        index=True,
    )
    source_youtube_id = Column(String(32), nullable=False)
    decision = Column(String(20), nullable=False)
    proposed_leaf_key = Column(String(100))
    reviewed_leaf_key = Column(String(100))
    transcript_quality = Column(String(30))
    reviewer = Column(String(255), nullable=False)
    notes = Column(Text)
    review_artifact_sha256 = Column(String(64), nullable=False)
    reviewed_at = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    collection_run = relationship("DatasetCollectionRun", back_populates="review_events")
    dataset = relationship("DatasetContent", back_populates="review_events")


class ClassificationModel(Base):
    __tablename__ = "classification_models"
    __table_args__ = (
        UniqueConstraint("model_key", "model_version", name="uq_classification_model_key_version"),
        Index("ix_classification_models_active", "is_active", "status"),
    )

    model_id = Column(Integer, primary_key=True)
    model_key = Column(String(100), nullable=False)
    model_version = Column(String(100), nullable=False)
    taxonomy_version = Column(String(50), nullable=False)
    model_type = Column(String(100), nullable=False)
    artifact_path = Column(String(1024))
    training_dataset_source = Column(String(100))
    training_dataset_version = Column(String(100))
    training_sample_count = Column(Integer, nullable=False, default=0)
    status = Column(String(30), nullable=False, default="draft")
    is_active = Column(Boolean, nullable=False, default=False)
    trained_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    evaluation_metrics = relationship(
        "ModelEvaluationMetric",
        back_populates="model",
        cascade="all, delete-orphan",
    )
    analysis_results = relationship("AnalysisResult", back_populates="classification_model")


class ModelEvaluationMetric(Base):
    __tablename__ = "model_evaluation_metrics"
    __table_args__ = (
        UniqueConstraint(
            "model_id",
            "dataset_split",
            "language",
            "taxonomy_level",
            "taxonomy_leaf_key",
            "metric_name",
            name="uq_model_evaluation_metric",
        ),
    )

    metric_id = Column(Integer, primary_key=True)
    model_id = Column(
        Integer,
        ForeignKey("classification_models.model_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    dataset_split = Column(String(20), nullable=False, default="test")
    language = Column(String(20), nullable=False, default="und")
    taxonomy_level = Column(Integer, nullable=False)
    taxonomy_leaf_key = Column(String(100), nullable=False, default="__overall__")
    metric_name = Column(String(50), nullable=False)
    metric_value = Column(Float, nullable=False)
    sample_size = Column(Integer, nullable=False, default=0)
    details = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    model = relationship("ClassificationModel", back_populates="evaluation_metrics")


class TrendingItem(Base):
    __tablename__ = "trending_items"

    item_id = Column(Integer, primary_key=True)
    keyword = Column(String(255), nullable=False, index=True)
    score = Column(Float, nullable=False, default=0.0)
    source = Column(String(100), nullable=False, default="unknown")
    domain = Column(String(100), nullable=True)
    fetched_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    meta = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    result_id = Column(Integer, primary_key=True)
    content_id = Column(Integer, ForeignKey("user_contents.content_id"), nullable=False)
    cluster_id = Column(Integer, ForeignKey("clusters.cluster_id"))
    dataset_id = Column(Integer, ForeignKey("dataset_contents.dataset_id"))
    classification_model_id = Column(Integer, ForeignKey("classification_models.model_id"))
    taxonomy_version = Column(String(50))
    taxonomy_leaf_key = Column(String(100))
    category_level_1 = Column(String(150))
    category_level_2 = Column(String(150))
    category_level_3 = Column(String(150))
    classification_confidence = Column(Float)
    classification_is_unknown = Column(Boolean, nullable=False, default=False)
    summary = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    content = relationship("UserContent", back_populates="analysis_results")
    cluster = relationship("Cluster", back_populates="analysis_results")
    dataset = relationship("DatasetContent", back_populates="analysis_results")
    classification_model = relationship("ClassificationModel", back_populates="analysis_results")


class ClusterRun(Base):
    __tablename__ = "cluster_runs"

    run_id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.user_id"))
    algorithm = Column(String(50), nullable=False, default="kmeans")
    n_clusters = Column(Integer, nullable=False)
    feature_dimension = Column(Integer, nullable=False, default=0)
    inertia = Column(Float, nullable=False, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    memberships = relationship("ClusterMembership", back_populates="run", cascade="all, delete-orphan")


class ClusterMembership(Base):
    __tablename__ = "cluster_memberships"

    membership_id = Column(Integer, primary_key=True)
    run_id = Column(Integer, ForeignKey("cluster_runs.run_id"), nullable=False)
    cluster_id = Column(Integer, ForeignKey("clusters.cluster_id"), nullable=False)
    content_id = Column(Integer, ForeignKey("user_contents.content_id"))
    dataset_id = Column(Integer, ForeignKey("dataset_contents.dataset_id"))
    item_text = Column(Text, nullable=False)
    top_terms = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    run = relationship("ClusterRun", back_populates="memberships")
    cluster = relationship("Cluster", back_populates="memberships")
    dataset = relationship("DatasetContent", back_populates="memberships")


class Recommendation(Base):
    __tablename__ = "recommendations"

    rec_id = Column(Integer, primary_key=True)
    content_id = Column(Integer, ForeignKey("user_contents.content_id"), nullable=False)
    recommended_keywords = Column(Text)
    recommended_duration = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    content = relationship("UserContent", back_populates="recommendations")


class SystemConfig(Base):
    __tablename__ = "system_configs"

    config_id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.user_id"))
    max_keywords = Column(Integer, nullable=False, default=10)
    hook_duration = Column(Integer, nullable=False, default=60)
    process_interval = Column(Integer, nullable=False, default=24)
    notification_batch_size = Column(Integer, nullable=False, default=50)
    youtube_region = Column(String(2), nullable=False, default="TH")
    google_region = Column(String(2), nullable=False, default="TH")
    tiktok_region = Column(String(2), nullable=False, default="TH")
    enable_youtube_trending = Column(Boolean, nullable=False, default=True)
    enable_google_trends = Column(Boolean, nullable=False, default=True)
    enable_tiktok_trending = Column(Boolean, nullable=False, default=True)
    auto_scan_interval_hours = Column(Integer, nullable=False, default=6)
    # New runtime/admin-configurable fields
    asr_model_default = Column(String(20), nullable=False, default="small")
    enable_model_toggle = Column(Boolean, nullable=False, default=True)
    job_backend = Column(String(20), nullable=False, default='inprocess')
    redis_url = Column(String(255), nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="configs")


class SystemLog(Base):
    __tablename__ = "system_logs"

    log_id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.user_id"))
    action = Column(String(255), nullable=False)
    status = Column(String(50), nullable=False)
    detail = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="logs")


class Notification(Base):
    __tablename__ = "notifications"
    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "watch_session_id",
            "trend_key",
            name="uq_notification_user_session_trend",
        ),
    )

    notification_id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False)
    watch_session_id = Column(
        Integer,
        ForeignKey("user_trend_watch_sessions.watch_session_id", ondelete="CASCADE"),
        nullable=False,
    )
    type = Column(String(50), nullable=False, default="new_live_trend")
    trend_key = Column(String(40), nullable=False)
    platform = Column(String(50), nullable=False)
    title = Column(String(500), nullable=False)
    category = Column(String(100), nullable=False, default="general")
    detected_at = Column(DateTime, nullable=False)
    payload = Column(Text)
    is_read = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="notifications")
    watch_session = relationship("UserTrendWatchSession", back_populates="notifications")


class FollowedTopic(Base):
    __tablename__ = "followed_topics"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False)
    match_type = Column(String(20), nullable=False)  # 'domain' or 'keyword'
    value = Column(String(255), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="followed_topics")


class UserTrendWatchSession(Base):
    __tablename__ = "user_trend_watch_sessions"

    watch_session_id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False)
    session_key = Column(String(64), nullable=False, unique=True, index=True)
    baseline_run_id = Column(
        Integer,
        ForeignKey("trend_snapshot_runs.run_id", ondelete="SET NULL"),
    )
    last_seen_run_id = Column(
        Integer,
        ForeignKey("trend_snapshot_runs.run_id", ondelete="SET NULL"),
    )
    is_active = Column(Boolean, nullable=False, default=True)
    started_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_seen_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    ended_at = Column(DateTime)

    user = relationship("User", back_populates="trend_watch_sessions")
    notifications = relationship(
        "Notification",
        back_populates="watch_session",
        cascade="all, delete-orphan",
    )


class TrendSnapshotRun(Base):
    __tablename__ = "trend_snapshot_runs"

    run_id = Column(Integer, primary_key=True)
    region = Column(String(10), nullable=False, default="TH", index=True)
    status = Column(String(20), nullable=False, default="running", index=True)
    provider_status = Column(Text, nullable=False, default="{}")
    total_items = Column(Integer, nullable=False, default=0)
    started_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime)

    items = relationship(
        "TrendSnapshotItem",
        back_populates="run",
        cascade="all, delete-orphan",
    )


class TrendSnapshotItem(Base):
    __tablename__ = "trend_snapshot_items"
    __table_args__ = (
        UniqueConstraint(
            "run_id",
            "platform",
            "trend_key",
            name="uq_trend_snapshot_item_run_platform_key",
        ),
        Index(
            "ix_trend_snapshot_platform_metric",
            "platform",
            "view_metric_version",
        ),
    )

    item_id = Column(Integer, primary_key=True)
    run_id = Column(Integer, ForeignKey("trend_snapshot_runs.run_id"), nullable=False, index=True)
    platform = Column(String(50), nullable=False, index=True)
    trend_key = Column(String(40), nullable=False)
    title = Column(String(500), nullable=False)
    category = Column(String(100), nullable=False, default="general")
    source_platform = Column(String(100), nullable=False)
    video_url = Column(String(1024))
    views = Column(Integer, nullable=False, default=0)
    likes = Column(Integer, nullable=False, default=0)
    comments = Column(Integer, nullable=False, default=0)
    trend_score = Column(Float, nullable=False, default=0.0)
    engagement_signal = Column(Float, nullable=False, default=0.0)
    view_metric_version = Column(
        String(64),
        nullable=False,
        default="unknown_v1",
    )
    published_at = Column(String(64))
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    run = relationship("TrendSnapshotRun", back_populates="items")
