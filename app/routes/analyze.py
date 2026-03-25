from fastapi import APIRouter, UploadFile, File
import shutil
import os

from app.services.ai_pipeline import analyze_video

router = APIRouter()


@router.post("/analyze")
async def analyze(file: UploadFile = File(...)):

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

    return result