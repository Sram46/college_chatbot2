import os
import re
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PDF_DIR = DATA_DIR / "pdfs"
DB_PATH = DATA_DIR / "index.db"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
_EMBED_MODEL = None


def ensure_data_dirs():
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_db_connection():
    ensure_data_dirs()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS colleges (
            id INTEGER PRIMARY KEY,
            name TEXT UNIQUE,
            pdf_path TEXT,
            uploaded_at TEXT
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY,
            college_id INTEGER,
            page_number INTEGER,
            text TEXT,
            embedding BLOB,
            created_at TEXT,
            FOREIGN KEY(college_id) REFERENCES colleges(id)
        )
        """
    )
    conn.commit()
    conn.close()


def get_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        _EMBED_MODEL = SentenceTransformer(MODEL_NAME)
    return _EMBED_MODEL


def extract_text_from_pdf(pdf_path):
    reader = PdfReader(str(pdf_path))
    pages = []
    for p, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append((p, text.strip()))
    return pages


def chunk_text(text):
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + CHUNK_SIZE, length)
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def embed_texts(texts):
    model = get_model()
    return model.encode(texts, normalize_embeddings=True)


def save_college(name, pdf_path):
    conn = get_db_connection()
    cursor = conn.cursor()
    now = datetime.utcnow().isoformat()
    cursor.execute("SELECT id FROM colleges WHERE name = ?", (name,))
    row = cursor.fetchone()
    if row:
        college_id = row["id"]
        cursor.execute(
            "UPDATE colleges SET pdf_path = ?, uploaded_at = ? WHERE id = ?",
            (str(pdf_path), now, college_id),
        )
        cursor.execute("DELETE FROM chunks WHERE college_id = ?", (college_id,))
    else:
        cursor.execute(
            "INSERT INTO colleges (name, pdf_path, uploaded_at) VALUES (?, ?, ?)",
            (name, str(pdf_path), now),
        )
        college_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return college_id


def save_chunks(college_id, page_number, texts, embeddings):
    conn = get_db_connection()
    cursor = conn.cursor()
    now = datetime.utcnow().isoformat()
    for text, vector in zip(texts, embeddings):
        embedding_blob = vector.astype(np.float32).tobytes()
        cursor.execute(
            "INSERT INTO chunks (college_id, page_number, text, embedding, created_at) VALUES (?, ?, ?, ?, ?)",
            (college_id, page_number, text, embedding_blob, now),
        )
    conn.commit()
    conn.close()


def ingest_pdf(source_path, name=None):
    init_db()
    ensure_data_dirs()
    source_path = Path(source_path)
    if name is None:
        name = source_path.stem
    target_path = PDF_DIR / source_path.name
    if source_path.resolve() != target_path.resolve():
        target_path.write_bytes(source_path.read_bytes())
    pages = extract_text_from_pdf(target_path)
    college_id = save_college(name, target_path)
    for page_number, text in pages:
        chunks = chunk_text(text)
        if not chunks:
            continue
        embeddings = embed_texts(chunks)
        save_chunks(college_id, page_number, chunks, embeddings)
    return college_id


def fetch_colleges():
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, pdf_path, uploaded_at FROM colleges ORDER BY name")
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def fetch_chunks():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT c.id, c.college_id, c.page_number, c.text, c.embedding, b.name AS college_name FROM chunks c JOIN colleges b ON c.college_id = b.id"
    )
    rows = cursor.fetchall()
    conn.close()
    return rows


def load_embeddings_and_metadata():
    rows = fetch_chunks()
    texts = []
    metadata = []
    embeddings = []
    for row in rows:
        texts.append(row["text"])
        metadata.append({
            "chunk_id": row["id"],
            "college_id": row["college_id"],
            "college_name": row["college_name"],
            "page_number": row["page_number"],
        })
        vector = np.frombuffer(row["embedding"], dtype=np.float32)
        embeddings.append(vector)
    if embeddings:
        return np.vstack(embeddings), metadata, texts
    return np.zeros((0, 384), dtype=np.float32), metadata, texts


def similarity_search(query, top_k=5):
    if top_k <= 0:
        return []
    model = get_model()
    query_vec = model.encode([query], normalize_embeddings=True)[0]
    vectors, metadata, texts = load_embeddings_and_metadata()
    if vectors.shape[0] == 0:
        return []
    scores = np.dot(vectors, query_vec)
    indices = np.argsort(scores)[::-1][:top_k]
    results = []
    for idx in indices:
        item = metadata[idx].copy()
        item["score"] = float(scores[idx])
        item["text"] = texts[idx]
        results.append(item)
    return results


def extract_facts_from_text(text):
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    facts = []
    patterns = [r"\b(seats?|capacity)\b", r"\b(fees?|fee)\b", r"\b(location|address)\b", r"\b(computer science|cs|engineering|management)\b"]
    for line in lines:
        for pattern in patterns:
            if re.search(pattern, line, flags=re.IGNORECASE):
                facts.append(line)
                break
    return facts


def answer_for_query(query, top_k=5):
    results = similarity_search(query, top_k=top_k)
    if not results:
        return "No college data available yet. Please upload college admission PDFs first."
    text = "\n\n".join([f"[{item['college_name']} - page {item['page_number']}]: {item['text']}" for item in results])
    low = query.lower()
    if any(keyword in low for keyword in ["compare", "vs", "versus", "better", "which college"]):
        return compare_colleges(query, results)
    if any(keyword in low for keyword in ["total", "sum", "all colleges", "overall"]):
        total_response = aggregate_numbers(query, results)
        if total_response:
            return total_response
    snippets = []
    seen = set()
    for item in results:
        facts = extract_facts_from_text(item["text"])
        if facts:
            snippets.append(f"{item['college_name']} (page {item['page_number']}): {facts[0]}")
        elif item["text"] not in seen:
            snippets.append(f"{item['college_name']} (page {item['page_number']}): {item['text'][:400]}")
            seen.add(item["text"])
    answer = "\n".join(snippets)
    if not answer:
        answer = text
    return answer


def compare_colleges(query, results):
    grouped = {}
    for item in results:
        college = item["college_name"]
        grouped.setdefault(college, []).extend(extract_facts_from_text(item["text"]))
    lines = [f"Comparison results for: {query}"]
    for college, facts in grouped.items():
        lines.append(f"\n{college}:")
        if facts:
            for fact in facts[:4]:
                lines.append(f" - {fact}")
        else:
            lines.append(" - No direct comparison facts found in the closest pages.")
    return "\n".join(lines)


def aggregate_numbers(query, results):
    low = query.lower()
    department = None
    match = re.search(r"computer science|cs|mechanical|civil|electrical|electronics|management|mba", low)
    if match:
        department = match.group(0)
    total = 0
    found = []
    for item in results:
        for num_match in re.finditer(r"(\d{1,5})\s*(?:seats|seat|capacity|intake)", item["text"], flags=re.IGNORECASE):
            value = int(num_match.group(1))
            context = item["text"][max(0, num_match.start() - 80):num_match.end() + 80]
            if department is None or re.search(re.escape(department), context, flags=re.IGNORECASE):
                total += value
                found.append((item["college_name"], value, context.strip()))
    if not found:
        return "I could not find matching seat counts in the closest college pages. Please try a more specific question or upload more detailed PDFs."
    lines = [f"Total {department or 'seat'} count across matching colleges: {total}"]
    for college, value, context in found:
        lines.append(f" - {college}: {value} ({context[:120]})")
    return "\n".join(lines)
