import os
import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.api.deps import require_roles
from app.database.db import SessionLocal, get_db
from sqlalchemy.orm import Session
from app.database.models import User
from app.services.analysis_settings import capture_analysis_settings, get_analysis_settings
from app.services.ai_pipeline import analyze_video as pipeline_analyze
from app.services.classification import classify_text_domain
from app.services.jobs import enqueue, update_current_job
from app.services.media_validation import (
    MediaValidationError,
    validate_user_upload_duration,
)
from app.services.nlp import normalize_text_for_nlp
from app.services.persistence import save_video_analysis_result
from app.services.recommendation import (
    build_classified_user_signal_snapshot,
    build_recommendation_from_analysis_data,
)
from app.services.taxonomy import normalize_taxonomy_leaf

router = APIRouter()


def _save_upload(file: UploadFile) -> str:
    os.makedirs("videos", exist_ok=True)
    safe_name = Path(file.filename or "upload.mp4").name
    file_path = os.path.join("videos", f"{uuid.uuid4().hex}_{safe_name}")
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return file_path


def _save_validated_upload(file: UploadFile, *, max_duration_seconds: int = 300) -> str:
    file_path = _save_upload(file)
    try:
        validate_user_upload_duration(file_path, max_duration_seconds=max_duration_seconds)
    except MediaValidationError as exc:
        Path(file_path).unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return file_path


def _build_recommendation(db, *, filename: str, result: dict, settings_snapshot: dict | None = None) -> tuple[dict, dict]:
    update_current_job(stage="classifying", progress=62, message="Classifying clip type")
    raw_transcript = str(
        result.get("raw_transcript")
        or result.get("transcript")
        or ""
    )
    cleaned_transcript = str(
        result.get("cleaned_transcript")
        or normalize_text_for_nlp(raw_transcript)
    )
    result["raw_transcript"] = raw_transcript
    result["cleaned_transcript"] = cleaned_transcript
    analysis = result.get("analysis", {})
    stt_meta = analysis.get("stt_meta", {})
    filename_fallback = stt_meta.get("transcript_source") == "fallback_filename"
    classification_transcript = "" if filename_fallback else cleaned_transcript
    hook_transcript = (
        ""
        if filename_fallback
        else str(
            analysis.get("hook_cleaned_transcript")
            or analysis.get("hook_transcript")
            or ""
        )
    )
    classification = classify_text_domain(
        db,
        title=None,
        text=classification_transcript,
        source_prefix="youtube",
        profile_limit=80,
        require_active_model=True,
        **({"model_snapshot": settings_snapshot["classification_model"]} if settings_snapshot else {}),
    )
    if filename_fallback:
        classification["input_source"] = "filename_fallback"
        classification["warning"] = stt_meta.get("warning") or (
            "Speech-to-text failed; the filename was not used for classification."
        )
    selected_domain = normalize_taxonomy_leaf(
        str(classification.get("taxonomy_leaf_key") or classification.get("domain"))
    )
    user_signals = build_classified_user_signal_snapshot(
        text=classification_transcript,
        hook_text=hook_transcript,
        taxonomy_leaf_key=selected_domain,
        max_keywords=10,
    )
    nlp_result = user_signals["nlp_result"]
    user_keywords = list(user_signals["user_keywords"])
    hook_terms = list(user_signals["hook_terms"])

    for legacy_key in (
        "product",
        "features",
        "entity_keywords",
        "context_keywords",
        "analysis_quality",
    ):
        analysis.pop(legacy_key, None)
    analysis["domain"] = selected_domain
    analysis["domain_source"] = str(classification.get("method") or "unknown")
    analysis["taxonomy_leaf_key"] = selected_domain
    analysis["category_level_1"] = classification.get("category_level_1")
    analysis["category_level_2"] = classification.get("category_level_2")
    analysis["category_level_3"] = classification.get("category_level_3")
    analysis["classification_confidence"] = float(
        classification.get("confidence") or 0.0
    )
    analysis["top_keywords"] = list(nlp_result.get("top_keywords", []))
    analysis["all_keywords"] = list(user_signals["content_keywords"])
    analysis["content_keywords"] = list(user_signals["content_keywords"])
    # Hook terms are observed in the uploaded clip; hook keywords are suggestions.
    analysis.pop("hook_keywords", None)
    analysis["hook_terms"] = list(user_signals["hook_terms"])
    analysis["comparable_keywords"] = list(user_signals["comparable_keywords"])
    analysis["comparable_keyword_evidence"] = list(
        user_signals["comparable_keyword_evidence"]
    )
    analysis["comparison_dimensions"] = list(
        user_signals["comparison_dimensions"]
    )
    analysis["dimension_status"] = list(user_signals["dimension_status"])

    update_current_job(stage="recommending", progress=76, message="Comparing with high-engagement dataset")
    recommendation = build_recommendation_from_analysis_data(
        db,
        domain=selected_domain,
        user_keywords=user_keywords,
        dimension_status=list(user_signals["dimension_status"]),
        hook_terms=hook_terms,
        transcript=classification_transcript,
        source_prefix="youtube",
        profile_limit=80,
    )
    recommendation["classification"] = classification
    recommendation["content_keywords"] = list(user_signals["content_keywords"])[:12]
    recommendation["hook_terms"] = hook_terms[:8]
    recommendation["comparable_keywords"] = list(
        user_signals["comparable_keywords"]
    )
    recommendation["comparable_keyword_evidence"] = list(
        user_signals["comparable_keyword_evidence"]
    )
    recommendation["keyword_sets"] = {
        "content": recommendation["content_keywords"],
        "hook": recommendation["hook_terms"],
        "comparable": recommendation["comparable_keywords"],
    }
    if isinstance(recommendation.get("evidence"), dict):
        recommendation["evidence"]["transcript_source"] = stt_meta.get("transcript_source") or "unknown"
        recommendation["evidence"]["transcript_scope"] = stt_meta.get("transcript_scope") or "unknown"
        recommendation["evidence"]["hook_seconds_analyzed"] = stt_meta.get("hook_seconds_analyzed")
        recommendation["evidence"]["stt_fallback_reason"] = stt_meta.get("fallback_reason")
        recommendation["evidence"]["warning"] = stt_meta.get("warning")
    return recommendation, nlp_result


def analyze_video_job(file_path: str, filename: str, user_id: int | None = None, *, settings_snapshot: dict | None = None) -> dict:
    db = SessionLocal()
    try:
        settings_snapshot = settings_snapshot or capture_analysis_settings(db)
        update_current_job(stage="extracting_audio", progress=18, message="Preparing full video audio")
        result = pipeline_analyze(
            file_path,
            display_name=filename,
            hook_duration_seconds=settings_snapshot["hook_duration_seconds"],
            asr_model_size=settings_snapshot["asr_model"],
        )
        result["analysis_settings"] = settings_snapshot
        recommendation, _nlp_result = _build_recommendation(db, filename=filename, result=result, settings_snapshot=settings_snapshot)
        result["recommendation"] = recommendation
        return result
    finally:
        db.close()


def analyze_and_save_video_job(file_path: str, filename: str, user_id: int, *, settings_snapshot: dict | None = None) -> dict:
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.user_id == user_id).first()
        if user is None:
            raise RuntimeError("User not found for analysis job.")

        settings_snapshot = settings_snapshot or capture_analysis_settings(db)
        update_current_job(stage="extracting_audio", progress=18, message="Preparing full video audio")
        result = pipeline_analyze(
            file_path,
            display_name=filename,
            hook_duration_seconds=settings_snapshot["hook_duration_seconds"],
            asr_model_size=settings_snapshot["asr_model"],
        )
        result["analysis_settings"] = settings_snapshot
        transcript = str(result.get("transcript") or "")
        raw_transcript = str(result.get("raw_transcript") or transcript)
        cleaned_transcript = str(
            result.get("cleaned_transcript")
            or normalize_text_for_nlp(raw_transcript)
        )
        recommendation, nlp_result = _build_recommendation(db, filename=filename, result=result, settings_snapshot=settings_snapshot)
        update_current_job(stage="saving", progress=90, message="Saving analysis to My Ideas")
        saved = save_video_analysis_result(
            db,
            user=user,
            filename=filename,
            file_path=file_path,
            transcript=transcript,
            raw_transcript=raw_transcript,
            cleaned_transcript=cleaned_transcript,
            analysis_payload=result,
            nlp_result=nlp_result,
            recommendation_payload=recommendation,
        )
        return {
            "content_id": saved["content_id"],
            "title": result.get("analysis", {}).get("title") or os.path.splitext(filename)[0],
            "transcript": transcript,
            "raw_transcript": raw_transcript,
            "cleaned_transcript": cleaned_transcript,
            "saved": True,
            "saved_keywords": saved["saved_keywords"],
            "recommended_keywords": saved["recommended_keywords"],
            "recommended_duration": saved["recommended_duration"],
            "recommendation": recommendation,
            "analysis": result,
            "analysis_settings": settings_snapshot,
            "nlp_result": nlp_result,
        }
    finally:
        db.close()


@router.get("/analyze/settings")
def read_upload_settings(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_roles("admin", "user")),
):
    return get_analysis_settings(db)


def _capture_upload_settings(db: Session) -> dict:
    try:
        return capture_analysis_settings(db)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    current_user: User = Depends(require_roles("admin", "user")),
    db: Session = Depends(get_db),
):
    print(f"[analyze] received upload: {file.filename}", flush=True)
    settings_snapshot = _capture_upload_settings(db)
    file_path = _save_validated_upload(file, max_duration_seconds=settings_snapshot["upload_max_duration_seconds"])
    filename = Path(file.filename or file_path).name
    job_id = enqueue(analyze_video_job, file_path, filename, current_user.user_id, settings_snapshot=settings_snapshot)
    return {"job_id": job_id}


@router.post("/analyze/save")
async def analyze_and_save(
    file: UploadFile = File(...),
    current_user: User = Depends(require_roles("admin", "user")),
    db: Session = Depends(get_db),
):
    print(f"[analyze/save] received upload: {file.filename}", flush=True)
    settings_snapshot = _capture_upload_settings(db)
    file_path = _save_validated_upload(file, max_duration_seconds=settings_snapshot["upload_max_duration_seconds"])
    filename = Path(file.filename or file_path).name
    print(f"[analyze/save] saved file to {file_path}, enqueueing analysis+save job", flush=True)
    job_id = enqueue(analyze_and_save_video_job, file_path, filename, current_user.user_id, settings_snapshot=settings_snapshot)
    return {"job_id": job_id}
