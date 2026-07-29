from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.services.jobs import get_status

router = APIRouter()


@router.get("/jobs/{job_id}")
def job_status(job_id: str):
    s = get_status(job_id)
    return JSONResponse(content=s)
