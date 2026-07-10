import os
import shutil

from fastapi import APIRouter, Depends, File, UploadFile
from sqlalchemy.orm import Session

from app.api.deps import require_roles
from app.database.db import get_db
from app.database.models import User
from app.services.ai_pipeline import analyze_video
from app.services.classification import classify_text_domain
from app.services.nlp import run_nlp_pipeline
from app.services.persistence import save_video_analysis_result
from app.services.recommendation import build_recommendation_from_analysis_data

router = APIRouter()


@router.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    _current_user: User = Depends(require_roles("admin", "user")),
    db: Session = Depends(get_db),
):

    # =========================
    # 1. save file
    # =========================
    os.makedirs("videos", exist_ok=True)
    file_path = f"videos/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # =========================
    # 2. run AI ONLY
    # =========================
    result = analyze_video(file_path)
    transcript = str(result.get("transcript") or "")
    analysis = result.get("analysis", {})
    classification = classify_text_domain(
        db,
        title=file.filename,
        text=transcript or file.filename,
        source_prefix="youtube",
        profile_limit=150,
    )
    selected_domain = str(analysis.get("domain") or "general")
    if float(classification.get("confidence", 0.0)) >= 0.45:
        selected_domain = str(classification["domain"])
    hook_terms = run_nlp_pipeline(transcript or file.filename, 8).get("filtered_tokens", [])[:8]
    recommendation = build_recommendation_from_analysis_data(
        db,
        domain=selected_domain,
        user_keywords=[item["keyword"] for item in analysis.get("top_keywords", [])],
        dimension_status=analysis.get("dimension_status", []),
        hook_terms=hook_terms,
        source_prefix="youtube",
        profile_limit=150,
    )
    recommendation["classification"] = classification
    result["recommendation"] = recommendation

    return result


@router.post("/analyze/save")
async def analyze_and_save(
    file: UploadFile = File(...),
    current_user: User = Depends(require_roles("admin", "user")),
    db: Session = Depends(get_db),
):
    os.makedirs("videos", exist_ok=True)
    file_path = f"videos/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = analyze_video(file_path)
    transcript = str(result.get("transcript") or "")
    nlp_result = run_nlp_pipeline(transcript or file.filename, 10)
    analysis = result.get("analysis", {})
    classification = classify_text_domain(
        db,
        title=file.filename,
        text=transcript or file.filename,
        source_prefix="youtube",
        profile_limit=150,
    )
    selected_domain = str(analysis.get("domain") or "general")
    if float(classification.get("confidence", 0.0)) >= 0.45:
        selected_domain = str(classification["domain"])
    recommendation = build_recommendation_from_analysis_data(
        db,
        domain=selected_domain,
        user_keywords=[item["keyword"] for item in analysis.get("top_keywords", [])] or [item["keyword"] for item in nlp_result.get("top_keywords", [])],
        dimension_status=analysis.get("dimension_status", []),
        hook_terms=nlp_result.get("filtered_tokens", [])[:8],
        source_prefix="youtube",
        profile_limit=150,
    )
    recommendation["classification"] = classification
    saved = save_video_analysis_result(
        db,
        user=current_user,
        filename=file.filename,
        file_path=file_path,
        transcript=transcript,
        analysis_payload=result,
        nlp_result=nlp_result,
        recommendation_payload=recommendation,
    )
    return {
        "content_id": saved["content_id"],
        "saved_keywords": saved["saved_keywords"],
        "recommended_keywords": saved["recommended_keywords"],
        "recommended_duration": saved["recommended_duration"],
        "recommendation": recommendation,
        "analysis": result,
        "nlp_result": nlp_result,
    }
