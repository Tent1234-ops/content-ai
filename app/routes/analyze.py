import os
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.api.deps import require_roles
from app.database.db import SessionLocal
from app.database.models import User
from app.services.ai_pipeline import analyze_video as pipeline_analyze
from app.services.classification import classify_text_domain
from app.services.jobs import enqueue, update_current_job
from app.services.media_validation import (
    MediaValidationError,
    validate_user_upload_duration,
)
from app.services.nlp import run_nlp_pipeline
from app.services.persistence import save_video_analysis_result
from app.services.recommendation import build_recommendation_from_analysis_data

router = APIRouter()


def _save_upload(file: UploadFile) -> str:
    os.makedirs("videos", exist_ok=True)
    safe_name = Path(file.filename or "upload.mp4").name
    file_path = os.path.join("videos", f"{uuid.uuid4().hex}_{safe_name}")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return file_path


def _save_validated_upload(file: UploadFile) -> str:
    file_path = _save_upload(file)
    try:
        validate_user_upload_duration(file_path)
    except MediaValidationError as exc:
        Path(file_path).unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return file_path


def _build_recommendation(db, *, filename: str, result: dict) -> tuple[dict, dict]:
    update_current_job(stage="classifying", progress=62, message="Classifying clip type")
    transcript = str(result.get("transcript") or "")
    nlp_result = run_nlp_pipeline(transcript or filename, 10)
    analysis = result.get("analysis", {})
    classification = classify_text_domain(
        db,
        title=filename,
        text=transcript or filename,
        source_prefix="youtube",
        profile_limit=80,
    )
    stt_meta = analysis.get("stt_meta", {})
    filename_fallback = stt_meta.get("transcript_source") == "fallback_filename"
    if filename_fallback:
        classification["confidence"] = min(
            float(classification.get("confidence", 0.0)),
            0.25,
        )
        classification["input_source"] = "filename_fallback"
        classification["warning"] = stt_meta.get("warning") or (
            "Speech-to-text failed; classification is based only on the filename."
        )
    selected_domain = str(analysis.get("domain") or "general")
    if classification.get("is_unknown"):
        selected_domain = "general"
    elif float(classification.get("confidence", 0.0)) >= 0.45:
        selected_domain = str(classification["domain"])

    user_keywords = [item["keyword"] for item in analysis.get("top_keywords", [])]
    if not user_keywords:
        user_keywords = [item["keyword"] for item in nlp_result.get("top_keywords", [])]

    update_current_job(stage="recommending", progress=76, message="Comparing with high-engagement dataset")
    recommendation = build_recommendation_from_analysis_data(
        db,
        domain=selected_domain,
        user_keywords=user_keywords,
        dimension_status=analysis.get("dimension_status", []),
        hook_terms=nlp_result.get("filtered_tokens", [])[:8],
        source_prefix="youtube",
        profile_limit=80,
    )
    recommendation["classification"] = classification
    recommendation["content_keywords"] = [
        item["keyword"] for item in nlp_result.get("top_keywords", [])
    ][:12]
    recommendation["hook_terms"] = nlp_result.get("filtered_tokens", [])[:8]
    if isinstance(recommendation.get("evidence"), dict):
        recommendation["evidence"]["transcript_source"] = stt_meta.get("transcript_source") or "unknown"
        recommendation["evidence"]["hook_seconds_analyzed"] = stt_meta.get("hook_seconds_analyzed")
        recommendation["evidence"]["stt_fallback_reason"] = stt_meta.get("fallback_reason")
        recommendation["evidence"]["warning"] = stt_meta.get("warning")
    return recommendation, nlp_result


def analyze_video_job(file_path: str, filename: str, user_id: int | None = None) -> dict:
    db = SessionLocal()
    try:
        update_current_job(stage="extracting_audio", progress=18, message="Extracting hook audio segment")
        result = pipeline_analyze(file_path, display_name=filename)
        recommendation, _nlp_result = _build_recommendation(db, filename=filename, result=result)
        result["recommendation"] = recommendation
        return result
    finally:
        db.close()


def analyze_and_save_video_job(file_path: str, filename: str, user_id: int) -> dict:
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.user_id == user_id).first()
        if user is None:
            raise RuntimeError("User not found for analysis job.")

        update_current_job(stage="extracting_audio", progress=18, message="Extracting hook audio segment")
        result = pipeline_analyze(file_path, display_name=filename)
        transcript = str(result.get("transcript") or "")
        recommendation, nlp_result = _build_recommendation(db, filename=filename, result=result)
        update_current_job(stage="saving", progress=90, message="Saving analysis to My Ideas")
        saved = save_video_analysis_result(
            db,
            user=user,
            filename=filename,
            file_path=file_path,
            transcript=transcript,
            analysis_payload=result,
            nlp_result=nlp_result,
            recommendation_payload=recommendation,
        )
        return {
            "content_id": saved["content_id"],
            "title": result.get("analysis", {}).get("title") or os.path.splitext(filename)[0],
            "transcript": transcript,
            "saved": True,
            "saved_keywords": saved["saved_keywords"],
            "recommended_keywords": saved["recommended_keywords"],
            "recommended_duration": saved["recommended_duration"],
            "recommendation": recommendation,
            "analysis": result,
            "nlp_result": nlp_result,
        }
    finally:
        db.close()


@router.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    current_user: User = Depends(require_roles("admin", "user")),
):
    print(f"[analyze] received upload: {file.filename}", flush=True)
    file_path = _save_validated_upload(file)
    filename = Path(file.filename or file_path).name
    job_id = enqueue(analyze_video_job, file_path, filename, current_user.user_id)
    return {"job_id": job_id}


@router.post("/analyze/save")
async def analyze_and_save(
    file: UploadFile = File(...),
    current_user: User = Depends(require_roles("admin", "user")),
):
    print(f"[analyze/save] received upload: {file.filename}", flush=True)
    file_path = _save_validated_upload(file)
    filename = Path(file.filename or file_path).name
    print(f"[analyze/save] saved file to {file_path}, enqueueing analysis+save job", flush=True)
    job_id = enqueue(analyze_and_save_video_job, file_path, filename, current_user.user_id)
    return {"job_id": job_id}
