from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text, text
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


class DatasetContent(Base):
    __tablename__ = "dataset_contents"

    dataset_id = Column(Integer, primary_key=True)
    title = Column(String(255), nullable=False)
    video_url = Column(String(1024))
    transcript = Column(Text)
    category = Column(String(100))
    source_platform = Column(String(50), nullable=False, default="youtube")
    views = Column(Integer, nullable=False, default=0)
    likes = Column(Integer, nullable=False, default=0)
    comments = Column(Integer, nullable=False, default=0)
    trend_score = Column(Float, nullable=False, default=0.0)
    duration_seconds = Column(Integer)
    published_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    analysis_results = relationship("AnalysisResult", back_populates="dataset")
    memberships = relationship("ClusterMembership", back_populates="dataset")


class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    result_id = Column(Integer, primary_key=True)
    content_id = Column(Integer, ForeignKey("user_contents.content_id"), nullable=False)
    cluster_id = Column(Integer, ForeignKey("clusters.cluster_id"))
    dataset_id = Column(Integer, ForeignKey("dataset_contents.dataset_id"))
    summary = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    content = relationship("UserContent", back_populates="analysis_results")
    cluster = relationship("Cluster", back_populates="analysis_results")
    dataset = relationship("DatasetContent", back_populates="analysis_results")


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
    asr_model_default = Column(String(20), nullable=False, default='small')
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

    notification_id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False)
    dataset_id = Column(Integer, ForeignKey("dataset_contents.dataset_id"), nullable=True)
    title = Column(String(255), nullable=False)
    body = Column(Text)
    link = Column(String(255))
    type = Column(String(50), nullable=False, default="system")
    payload = Column(Text)  # optional JSON payload
    message = Column(Text, nullable=False, default="", server_default=text("''"))
    topic = Column(String(100), nullable=False, default="general", server_default=text("'general'"))
    source_platform = Column(String(50), nullable=False, default="system", server_default=text("'system'"))
    trend_score = Column(Float, nullable=False, default=0.0, server_default=text("0.0"))
    is_read = Column(Boolean, nullable=False, default=False)
    delivered_via_ws = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="notifications")


class FollowedTopic(Base):
    __tablename__ = "followed_topics"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False)
    match_type = Column(String(20), nullable=False)  # 'domain' or 'keyword'
    value = Column(String(255), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="followed_topics")
