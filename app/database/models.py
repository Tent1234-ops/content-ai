from sqlalchemy import Column, Integer, String, Text, ForeignKey, DateTime
from datetime import datetime
from .db import Base

class User(Base):
    __tablename__ = "users"
    user_id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100))
    email = Column(String(255))
    password = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)

class UserContent(Base):
    __tablename__ = "user_contents"
    content_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.user_id"))
    title = Column(String(255))
    video_url = Column(String(255))
    transcript = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

class Keyword(Base):
    __tablename__ = "keywords"
    keyword_id = Column(Integer, primary_key=True)
    keyword = Column(String(100), unique=True)

class ContentKeyword(Base):
    __tablename__ = "content_keywords"
    id = Column(Integer, primary_key=True)
    content_id = Column(Integer, ForeignKey("user_contents.content_id"))
    keyword_id = Column(Integer, ForeignKey("keywords.keyword_id"))

class AnalysisResult(Base):
    __tablename__ = "analysis_results"
    result_id = Column(Integer, primary_key=True)
    content_id = Column(Integer, ForeignKey("user_contents.content_id"))
    summary = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)