import sys
import os
import re
import json
import chromadb
from pathlib import Path
from datetime import datetime, timezone
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

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


def _get_paths(filename: str):
    """Return (pdf_path, md_path, tree_path) for a given filename stem."""
    stem = Path(filename).stem
    pdf_path  = UPLOAD_DIR / f"{stem}.pdf"
    md_path   = UPLOAD_DIR / f"{stem}.md"
    tree_path = UPLOAD_DIR / f"{stem}.tree.json"
    return pdf_path, md_path, tree_path


def _history_path(filename: str) -> Path:
    stem = Path(filename).stem
    return UPLOAD_DIR / f"{stem}.history.json"


def _load_history(filename: str) -> list:
    path = _history_path(filename)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("messages", [])
    except Exception:
        return []


def _append_history(filename: str, user_query: str, assistant_answer: str) -> None:
    path = _history_path(filename)
    messages = _load_history(filename)
    now = datetime.now(timezone.utc).isoformat()
    messages.append({"role": "user",      "content": user_query,        "timestamp": now})
    messages.append({"role": "assistant", "content": assistant_answer,  "timestamp": now})
    path.write_text(
        json.dumps({"filename": filename, "messages": messages}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    stem = Path(file.filename).stem
    pdf_path, md_path, tree_path = _get_paths(file.filename)

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

        collection = get_chroma_collection(stem)
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
async def get_status(filename: str):
    _, md_path, tree_path = _get_paths(filename)
    stem = Path(filename).stem

    if tree_path.exists():
        tree       = load_tree(tree_path)
        collection = get_chroma_collection(stem)
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
async def query_document(req: QueryRequest):
    _, _, tree_path = _get_paths(req.filename)
    stem = Path(req.filename).stem

    if not tree_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No tree found for '{req.filename}'. Please upload the PDF first."
        )

    try:
        tree       = load_tree(tree_path)
        collection = get_chroma_collection(stem)
        answer     = retrieve_and_answer(tree, req.query, collection)

        # Persist this exchange to history
        _append_history(req.filename, req.query, answer)

        return {
            "answer":   answer,
            "query":    req.query,
            "filename": req.filename,
            "hybrid":   collection_is_indexed(collection),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@router.get("/history/{filename}")
async def get_history(filename: str):
    """Return the full conversation history for a document."""
    return {"filename": filename, "messages": _load_history(filename)}


@router.post("/reindex/{filename}")
async def reindex_document(filename: str):
    _, _, tree_path = _get_paths(filename)
    stem = Path(filename).stem

    if not tree_path.exists():
        raise HTTPException(status_code=404, detail=f"No tree found for '{filename}'.")

    try:
        tree   = load_tree(tree_path)
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        safe_name = re.sub(r"[^a-zA-Z0-9\-]", "-", stem)[:63]
        try:
            client.delete_collection(safe_name)
        except Exception:
            pass

        collection = get_chroma_collection(stem)
        build_chunk_index(tree, collection)

        return {"status": "reindexed", "filename": filename, "chunk_count": collection.count()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reindex failed: {str(e)}")


@router.get("/documents")
async def list_documents():
    docs = []
    for tree_path in UPLOAD_DIR.glob("*.tree.json"):
        stem = tree_path.stem.replace(".tree", "")
        try:
            tree       = load_tree(tree_path)
            collection = get_chroma_collection(stem)
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