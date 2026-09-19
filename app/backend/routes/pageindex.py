import sys
import os
import re
import json
import hashlib
import chromadb
from pathlib import Path
from datetime import datetime, timezone
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.backend.auth import get_device_id

# ---------------------------------------------------------------------------
# PAGEINDEX_DIR — where pageindex.py and ocr.py live inside the repo
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[3]
PAGEINDEX_DIR = Path(os.environ.get("PAGEINDEX_DIR", str(_REPO_ROOT / "pageocr")))
sys.path.insert(0, str(_REPO_ROOT))

from pageocr.ocr import pdf_to_text, text_to_markdown, save_markdown
from pageocr.pageindex import (
    build_tree, summarize_tree, save_tree, load_tree,
    retrieve_and_answer, clean_markdown,
    get_chroma_collection, build_chunk_index, collection_is_indexed,
)

router = APIRouter(prefix="/pageindex", tags=["pageindex"])

# ---------------------------------------------------------------------------
# STORAGE DIRS
# ---------------------------------------------------------------------------
UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", str(PAGEINDEX_DIR / "uploads")))
CHROMA_DIR = Path(os.environ.get("CHROMA_DIR", str(PAGEINDEX_DIR / "chroma_db")))

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)


class QueryRequest(BaseModel):
    filename: str
    query: str


def _device_prefix(device_id: str) -> str:
    """
    Short, stable prefix derived from device_id. Used to namespace Chroma
    collection names — Chroma caps names at 63 chars, so a full 36-char
    UUID would eat most of that budget and leave little room for the
    document name itself.
    """
    return hashlib.sha256(device_id.encode()).hexdigest()[:12]


def _device_dir(device_id: str) -> Path:
    """Each device gets its own upload subdirectory — this is what actually
    keeps devices from seeing or colliding with each other's documents."""
    d = UPLOAD_DIR / device_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _get_paths(filename: str, device_id: str):
    """Return (pdf_path, md_path, tree_path) for a given filename stem, scoped to device_id."""
    stem = Path(filename).stem
    device_dir = _device_dir(device_id)
    pdf_path  = device_dir / f"{stem}.pdf"
    md_path   = device_dir / f"{stem}.md"
    tree_path = device_dir / f"{stem}.tree.json"
    return pdf_path, md_path, tree_path


def _history_path(filename: str, device_id: str) -> Path:
    stem = Path(filename).stem
    return _device_dir(device_id) / f"{stem}.history.json"


def _load_history(filename: str, device_id: str) -> list:
    path = _history_path(filename, device_id)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("messages", [])
    except Exception:
        return []


def _append_history(filename: str, device_id: str, user_query: str, assistant_answer: str) -> None:
    path = _history_path(filename, device_id)
    messages = _load_history(filename, device_id)
    now = datetime.now(timezone.utc).isoformat()
    messages.append({"role": "user",      "content": user_query,        "timestamp": now})
    messages.append({"role": "assistant", "content": assistant_answer,  "timestamp": now})
    path.write_text(
        json.dumps({"filename": filename, "messages": messages}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _collection_key(stem: str, device_id: str) -> str:
    """Chroma collection name, namespaced per device so one device's
    documents are never retrievable by another's queries."""
    return f"{_device_prefix(device_id)}_{stem}"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...), device_id: str = Depends(get_device_id)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    stem = Path(file.filename).stem
    pdf_path, md_path, tree_path = _get_paths(file.filename, device_id)

    contents = await file.read()
    with open(pdf_path, "wb") as f:
        f.write(contents)

    try:
        raw_text = pdf_to_text(str(pdf_path))
        md_text  = text_to_markdown(raw_text)
        save_markdown(md_text, str(md_path))

        cleaned = clean_markdown(md_text)
        tree    = build_tree(cleaned)
        summarize_tree(tree)
        save_tree(tree, tree_path)

        collection = get_chroma_collection(_collection_key(stem, device_id))
        build_chunk_index(tree, collection)

        return JSONResponse({
            "status":      "ready",
            "filename":    file.filename,
            "stem":        stem,
            "node_count":  tree.node_count(),
            "chunk_count": collection.count(),
            "md_saved":    str(md_path),
            "tree_saved":  str(tree_path),
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@router.get("/status/{filename}")
async def get_status(filename: str, device_id: str = Depends(get_device_id)):
    _, md_path, tree_path = _get_paths(filename, device_id)
    stem = Path(filename).stem

    if tree_path.exists():
        tree       = load_tree(tree_path)
        collection = get_chroma_collection(_collection_key(stem, device_id))
        indexed    = collection_is_indexed(collection)
        return {
            "status":      "ready",
            "filename":    filename,
            "node_count":  tree.node_count(),
            "chunk_index": "ready" if indexed else "missing",
            "chunk_count": collection.count() if indexed else 0,
        }
    elif md_path.exists():
        return {"status": "md_only", "filename": filename}
    else:
        return {"status": "not_found", "filename": filename}


@router.post("/query")
async def query_document(req: QueryRequest, device_id: str = Depends(get_device_id)):
    _, _, tree_path = _get_paths(req.filename, device_id)
    stem = Path(req.filename).stem

    if not tree_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No tree found for '{req.filename}'. Please upload the PDF first."
        )

    try:
        tree       = load_tree(tree_path)
        collection = get_chroma_collection(_collection_key(stem, device_id))
        answer     = retrieve_and_answer(tree, req.query, collection)

        # Persist this exchange to history
        _append_history(req.filename, device_id, req.query, answer)

        return {
            "answer":   answer,
            "query":    req.query,
            "filename": req.filename,
            "hybrid":   collection_is_indexed(collection),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@router.get("/history/{filename}")
async def get_history(filename: str, device_id: str = Depends(get_device_id)):
    """Return the full conversation history for a document."""
    return {"filename": filename, "messages": _load_history(filename, device_id)}


@router.post("/reindex/{filename}")
async def reindex_document(filename: str, device_id: str = Depends(get_device_id)):
    _, _, tree_path = _get_paths(filename, device_id)
    stem = Path(filename).stem

    if not tree_path.exists():
        raise HTTPException(status_code=404, detail=f"No tree found for '{filename}'.")

    try:
        tree   = load_tree(tree_path)
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        safe_name = re.sub(r"[^a-zA-Z0-9\-]", "-", _collection_key(stem, device_id))[:63]
        try:
            client.delete_collection(safe_name)
        except Exception:
            pass

        collection = get_chroma_collection(_collection_key(stem, device_id))
        build_chunk_index(tree, collection)

        return {"status": "reindexed", "filename": filename, "chunk_count": collection.count()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reindex failed: {str(e)}")


@router.get("/documents")
async def list_documents(device_id: str = Depends(get_device_id)):
    docs = []
    device_dir = _device_dir(device_id)
    for tree_path in device_dir.glob("*.tree.json"):
        stem = tree_path.stem.replace(".tree", "")
        try:
            tree       = load_tree(tree_path)
            collection = get_chroma_collection(_collection_key(stem, device_id))
            indexed    = collection_is_indexed(collection)
            docs.append({
                "filename":    f"{stem}.pdf",
                "stem":        stem,
                "node_count":  tree.node_count(),
                "chunk_count": collection.count() if indexed else 0,
                "hybrid_ready": indexed,
                "status":      "ready",
            })
        except Exception:
            docs.append({"filename": f"{stem}.pdf", "stem": stem, "status": "error"})
    return {"documents": docs}