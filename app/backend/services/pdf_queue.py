"""Async PDF generation queue.

One background worker processes jobs one at a time.
Start the queue from FastAPI's startup_event by calling pdf_queue.start().
"""
import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class PDFJobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PDFJob:
    queue_id: str
    email: str
    run_data: Dict[str, Any]
    status: PDFJobStatus = PDFJobStatus.QUEUED
    created_at: datetime = field(default_factory=datetime.utcnow)
    error: Optional[str] = None


class PDFQueue:
    def __init__(self) -> None:
        self._jobs: Dict[str, PDFJob] = {}   # insertion-ordered (Python 3.7+)
        self._q: Optional[asyncio.Queue] = None
        self._worker_task: Optional[asyncio.Task] = None

    # ── lifecycle ──────────────────────────────────────────────────────────

    def start(self) -> None:
        """Call once from FastAPI startup_event."""
        self._q = asyncio.Queue()
        self._worker_task = asyncio.create_task(self._worker())

    # ── worker ─────────────────────────────────────────────────────────────

    async def _worker(self) -> None:
        while True:
            queue_id: str = await self._q.get()
            job = self._jobs.get(queue_id)

            if not job or job.status == PDFJobStatus.CANCELLED:
                self._q.task_done()
                continue

            job.status = PDFJobStatus.PROCESSING
            try:
                loop = asyncio.get_running_loop()

                # PDF generation is synchronous — run in thread pool
                from app.backend.services.pdf_service import generate_pdf
                pdf_bytes: bytes = await loop.run_in_executor(
                    None, generate_pdf, job.run_data
                )

                # Email is async-wrapped
                from app.backend.services.email_service import send_pdf_email
                await send_pdf_email(
                    to_email=job.email,
                    pdf_bytes=pdf_bytes,
                    flow_name=job.run_data.get("flow_name") or "AI Hedge Fund",
                )
                job.status = PDFJobStatus.DONE

            except Exception as exc:
                job.status = PDFJobStatus.FAILED
                job.error = str(exc)
                print(f"[pdf_queue] job {queue_id} failed: {exc}")
            finally:
                self._q.task_done()

    # ── public API ─────────────────────────────────────────────────────────

    async def add_job(self, email: str, run_data: Dict[str, Any]) -> PDFJob:
        """Add a new PDF job to the queue. Returns the job."""
        queue_id = str(uuid.uuid4())
        job = PDFJob(queue_id=queue_id, email=email, run_data=run_data)
        self._jobs[queue_id] = job
        await self._q.put(queue_id)
        return job

    def get_position(self, queue_id: str) -> int:
        """Return 1-based queue position for QUEUED jobs; 0 otherwise."""
        job = self._jobs.get(queue_id)
        if not job or job.status != PDFJobStatus.QUEUED:
            return 0
        pos = 1
        for j in self._jobs.values():
            if j.queue_id == queue_id:
                break
            if j.status == PDFJobStatus.QUEUED:
                pos += 1
        return pos

    def cancel_job(self, queue_id: str) -> bool:
        """Cancel a queued job. Returns True if successful."""
        job = self._jobs.get(queue_id)
        if job and job.status == PDFJobStatus.QUEUED:
            job.status = PDFJobStatus.CANCELLED
            return True
        return False

    def list_jobs(self) -> List[PDFJob]:
        """Return all non-cancelled jobs (active + recent done/failed)."""
        return [j for j in self._jobs.values() if j.status != PDFJobStatus.CANCELLED]


# Singleton — import this everywhere
pdf_queue = PDFQueue()