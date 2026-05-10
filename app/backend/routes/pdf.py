from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

from app.backend.services.pdf_queue import pdf_queue, PDFJob, PDFJobStatus

router = APIRouter(prefix="/pdf", tags=["pdf"])


# ── request / response models ─────────────────────────────────────────────────

class RunDataPayload(BaseModel):
    decisions: Dict[str, Any]
    analyst_signals: Dict[str, Any]
    current_prices: Dict[str, float]
    tickers: Optional[List[str]] = None
    flow_name: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class PDFRequest(BaseModel):
    email: str
    run_data: RunDataPayload


class PDFJobResponse(BaseModel):
    queue_id: str
    status: str
    email: str
    position: int
    created_at: str
    error: Optional[str] = None


def _to_response(job: PDFJob) -> PDFJobResponse:
    return PDFJobResponse(
        queue_id=job.queue_id,
        status=job.status.value,
        email=job.email,
        position=pdf_queue.get_position(job.queue_id),
        created_at=job.created_at.isoformat(),
        error=job.error,
    )


# ── endpoints ─────────────────────────────────────────────────────────────────

@router.post("/request")
async def request_pdf(body: PDFRequest):
    """Queue a PDF generation job. Returns queue_id and position."""
    if not body.email or "@" not in body.email:
        raise HTTPException(status_code=400, detail="Valid email required.")
    try:
        # Derive tickers from decisions if not provided
        run_data = body.run_data.model_dump()
        if not run_data.get("tickers"):
            run_data["tickers"] = list(run_data.get("decisions", {}).keys())

        job = await pdf_queue.add_job(email=body.email, run_data=run_data)
        return {
            "queue_id": job.queue_id,
            "position": pdf_queue.get_position(job.queue_id),
            "message": f"PDF queued — will be emailed to {body.email}.",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to queue PDF: {e}")


@router.get("/queue", response_model=List[PDFJobResponse])
async def get_queue():
    """Return all active and recent PDF jobs (excludes cancelled)."""
    return [_to_response(j) for j in pdf_queue.list_jobs()]


@router.delete("/queue/{queue_id}")
async def cancel_pdf(queue_id: str):
    """Cancel a queued PDF job. Only works while status is 'queued'."""
    success = pdf_queue.cancel_job(queue_id)
    if not success:
        raise HTTPException(
            status_code=400,
            detail="Job not found or already processing / completed.",
        )
    return {"message": "Job cancelled."}