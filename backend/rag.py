import os
import re
import pickle
from pathlib import Path

import numpy as np
import faiss
import torch
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

# ----------------------------
# CONFIGURATION
# ----------------------------
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent

PDF_DIR = PROJECT_ROOT / "data" / "pdfs"
INDEX_DIR = PROJECT_ROOT / "data" / "index"

EMBED_MODEL_NAME = r"D:\OneDrive - SOLIDA\pdf_rag\models\paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL_NAME = r"D:\OneDrive - SOLIDA\pdf_rag\models\phi-2"

CHUNK_SIZE = 700
CHUNK_OVERLAP = 150
TOP_K = 6

# ----------------------------
# LOAD MODELS
# ----------------------------
print("Loading embedding model...")
embedder = SentenceTransformer(EMBED_MODEL_NAME)

print("Loading LLM model locally...")
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME, device_map="cpu")
model.eval()

# ----------------------------
# PDF → TEXT
# ----------------------------
def _clean_text(t: str) -> str:
    if not t:
        return ""
    t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)            # dehyphenate
    t = re.sub(r"(?<!\n)\n(?!\n)", " ", t)            # single newlines -> space
    t = re.sub(r"[ \t]+", " ", t)                     # collapse whitespace
    return t.strip()

def pdf_to_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    pages = []
    for page in reader.pages:
        pages.append(_clean_text(page.extract_text() or ""))
    return "\n\n".join(p for p in pages if p)

# ----------------------------
# CHUNKING
# ----------------------------
def chunk_text(text: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks

# ----------------------------
# INDEX BUILD
# ----------------------------
def build_faiss_index(pdf_dir=PDF_DIR, index_dir=INDEX_DIR):
    os.makedirs(index_dir, exist_ok=True)

    all_chunks = []
    metadata = []

    print("Reading PDFs and creating chunks...")
    for filename in os.listdir(pdf_dir):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(pdf_dir, filename)
            text = pdf_to_text(pdf_path)
            chunks = chunk_text(text)
            all_chunks.extend(chunks)
            metadata.extend([filename] * len(chunks))

    print(f"Total chunks: {len(all_chunks)}")
    print("Creating embeddings...")
    embeddings = embedder.encode(
        all_chunks,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embeddings = np.asarray(embeddings, dtype="float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # cosine similarity when normalized
    index.add(embeddings)

    faiss.write_index(index, os.path.join(index_dir, "faiss_index.bin"))
    with open(os.path.join(index_dir, "metadata.pkl"), "wb") as f:
        pickle.dump((all_chunks, metadata), f)

    print("FAISS index and metadata saved!")

# ----------------------------
# RETRIEVE
# ----------------------------
def retrieve_chunks(query: str, index_dir=INDEX_DIR, top_k=TOP_K):
    index = faiss.read_index(os.path.join(index_dir, "faiss_index.bin"))
    with open(os.path.join(index_dir, "metadata.pkl"), "rb") as f:
        all_chunks, metadata = pickle.load(f)

    query_vec = embedder.encode([query], normalize_embeddings=True)
    query_vec = np.asarray(query_vec, dtype="float32")

    scores, indices = index.search(query_vec, top_k)

    retrieved_chunks = [all_chunks[i] for i in indices[0] if i != -1]
    retrieved_files = [metadata[i] for i in indices[0] if i != -1]
    return retrieved_chunks, retrieved_files

# ----------------------------
# SECTION EXTRACTOR (robust for “Señales de salida/entrada”)
# ----------------------------
def extract_section_from_all_chunks(heading: str, index_dir=INDEX_DIR, max_chars=4000):
    with open(os.path.join(index_dir, "metadata.pkl"), "rb") as f:
        all_chunks, metadata = pickle.load(f)

    key = heading.lower()
    best = None  # (score, snippet, src)

    for ch, src in zip(all_chunks, metadata):
        low = ch.lower()
        pos = low.find(key)
        if pos == -1:
            continue

        raw = ch[pos:pos + max_chars].strip()

        # Skip TOC-like matches: "SEÑALES DE SALIDA 14 4. PROYECTO DEMO..."
        if re.search(r"SEÑALES DE SALIDA\s+\d+\s+\d+\.", raw.upper()):
            continue

        # Cut at next numbered heading AFTER the first line
        # Examples: "3.2 ...", "4. Proyecto ...", "3.5.4 ..."
        m = re.search(r"\n\s*\d+(\.\d+)*\s+[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]", raw)
        if m:
            raw = raw[:m.start()].strip()

        # score preference: real numbered section near the start
        score = 0
        if re.search(r"^\s*\d+(\.\d+)*", raw):
            score += 5
        if "el bloque" in raw.lower():
            score += 1
        if "señal" in raw.lower():
            score += 1

        if best is None or score > best[0]:
            best = (score, raw, src)

    if best is None:
        return None, None
    return best[1], best[2]

# ----------------------------
# GENERATE
# ----------------------------
def generate_answer(query: str, retrieved_chunks, max_new_tokens=120):
    context = "\n\n".join(retrieved_chunks)
    prompt = f"""Use ONLY the context to answer.
If the answer is not in the context, reply exactly: I don't know.

Context:
{context}

Question:
{query}

Answer:"""

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to("cpu") for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    for stop in ["\nQuestion:", "\nContext:", "\nAnswer:"]:
        if stop in text:
            text = text.split(stop)[0].strip()

    return text

def format_signals_answer(section_text: str, src: str):
    # 1) Best case: signals listed inside parentheses after the phrase
    m = re.search(r"señales de salida\s*\(([^)]+)\)", section_text, flags=re.IGNORECASE)
    if m:
        inside = m.group(1)
        toks = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]{1,}\b", inside)
        # remove Spanish connectors like 'y' if present
        toks = [t for t in toks if t.lower() not in {"y", "o"}]
        uniq = list(dict.fromkeys(toks))
        if uniq:
            ans = "Señales de salida:\n" + "\n".join(f"- {s}" for s in uniq)
            return ans, [src]

    # 2) Next best: bullet lines (manuals often list signals as '- X ...')
    lines = [ln.strip() for ln in section_text.splitlines()]
    bullet_items = []
    for ln in lines:
        if ln.startswith("-") or ln.startswith("•"):
            ln = ln.lstrip("-•").strip()
            # take first token as the signal name
            name = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]{1,}\b", ln)
            if name:
                bullet_items.append(name[0])
    bullet_items = list(dict.fromkeys(bullet_items))
    if bullet_items:
        ans = "Señales de salida:\n" + "\n".join(f"- {s}" for s in bullet_items[:40])
        return ans, [src]

    return "I don't know", [src]
