# backend/ingest.py
from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
import shutil

from .rag import PDF_DIR, build_faiss_index

router = APIRouter()

@router.post("/ingest")
async def ingest_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    PDF_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(PDF_DIR) / file.filename

    # Save upload to disk (streaming-friendly)
    with out_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Rebuild index (simple + deterministic for MVP)
    build_faiss_index()

    return {"status": "ok", "saved_as": file.filename}
