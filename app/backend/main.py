from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(env_path, override=True)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import asyncio
import os

from app.backend.routes import api_router
from app.backend.database.connection import engine
from app.backend.database.models import Base
from app.backend.services.ollama_service import ollama_service

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Hedge Fund API", description="Backend API for AI Hedge Fund", version="0.1.0")

@app.get("/health")
def health():
    return {"status": "ok"}

# Initialize database tables (safe to run multiple times)
Base.metadata.create_all(bind=engine)

# Configure CORS
# Local dev origins are always allowed; production origins (droplet IP/domain)
# come from CORS_ORIGINS in .env as a comma-separated list.
_dev_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
]
_env_origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_dev_origins + _env_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include all routes
app.include_router(api_router)

from app.backend.routes.pageindex import router as pageindex_router
app.include_router(pageindex_router)


@app.on_event("startup")
async def startup_event():
    """Startup: check Ollama availability and start PDF queue worker."""

    # ── Ollama check ──────────────────────────────────────────────────────
    try:
        logger.info("Checking Ollama availability...")
        status = await ollama_service.check_ollama_status()

        if status["installed"]:
            if status["running"]:
                logger.info(f"✓ Ollama is installed and running at {status['server_url']}")
                if status["available_models"]:
                    logger.info(f"✓ Available models: {', '.join(status['available_models'])}")
                else:
                    logger.info("ℹ No models are currently downloaded")
            else:
                logger.info("ℹ Ollama is installed but not running")
        else:
            logger.info("ℹ Ollama is not installed. Visit https://ollama.com to install.")

    except Exception as e:
        logger.warning(f"Could not check Ollama status: {e}")

    # ── PDF queue worker ──────────────────────────────────────────────────
    try:
        from app.backend.services.pdf_queue import pdf_queue
        pdf_queue.start()
        logger.info("✓ PDF queue worker started")
    except Exception as e:
        logger.warning(f"Could not start PDF queue worker: {e}")