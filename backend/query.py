from fastapi import APIRouter
from pydantic import BaseModel
import re

from .rag import (
    retrieve_chunks,
    generate_answer,
    extract_section_from_all_chunks,
    format_signals_answer,   # <-- NEW
)

router = APIRouter()

class QueryRequest(BaseModel):
    question: str

def _looks_like_identifier_question(q: str) -> bool:
    q = q.lower()
    keys = [
        "variable", "named", "name", "called",
        "se llama", "nombre", "identificador",
        "parámetro", "parametro",
    ]
    return any(k in q for k in keys)

def _exact_candidates_from_chunks(chunks, max_items=40):
    ids = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b", "\n".join(chunks))
    out, seen = [], set()
    for x in ids:
        if x not in seen:
            seen.add(x)
            out.append(x)
        if len(out) >= max_items:
            break
    return out

@router.post("/query")
def query_rag(payload: QueryRequest):
    question = payload.question.strip()
    q_low = question.lower()

    # 1) Section/list questions -> bullets + sources
    if "señales de salida" in q_low or "senales de salida" in q_low:
        snippet, src = extract_section_from_all_chunks("Señales de salida")
        if snippet:
            ans, sources = format_signals_answer(snippet, src)
            return {"question": question, "answer": ans, "sources": sources}

    if "señales de entrada" in q_low or "senales de entrada" in q_low:
        snippet, src = extract_section_from_all_chunks("Señales de entrada")
        if snippet:
            ans, sources = format_signals_answer(snippet, src)
            return {"question": question, "answer": ans, "sources": sources}

    # 2) Normal retrieval
    chunks, files = retrieve_chunks(question)
    debug_chunks = True

    # 3) Identifier questions
    if _looks_like_identifier_question(question):
        cands = _exact_candidates_from_chunks(chunks)
        if cands:
            return {
                "question": question,
                "answer": "Candidate identifiers found in the most relevant text:\n" + "\n".join(cands),
                "sources": files,
                **({"chunks": chunks} if debug_chunks else {}),
            }

    # 4) Normal RAG
    answer = generate_answer(question, chunks)
    return {
        "question": question,
        "answer": answer,
        "sources": files,
        **({"chunks": chunks} if debug_chunks else {}),
    }
