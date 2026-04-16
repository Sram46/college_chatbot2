import os
import shutil
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import pdf_indexer

app = FastAPI()
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
TEMPLATES_DIR.mkdir(exist_ok=True)

pdf_indexer.init_db()
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>College Admission Chatbot</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f7f9fc; }
        .card { background: #fff; border-radius: 10px; box-shadow: 0 2px 12px rgba(0,0,0,0.08); margin: 20px auto; max-width: 900px; padding: 24px; }
        h1 { margin-top: 0; }
        input, textarea, button { width: 100%; font-size: 1rem; margin: 8px 0; padding: 12px; border: 1px solid #ccd4de; border-radius: 8px; }
        button { background: #2f72d5; color: #fff; cursor: pointer; border: none; }
        button:hover { opacity: 0.95; }
        .row { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .full { grid-column: 1 / -1; }
        .result { min-height: 220px; white-space: pre-wrap; }
    </style>
</head>
<body>
    <div class="card">
        <h1>College Admission Chatbot</h1>
        <p>Upload college admission PDFs and ask detailed questions about seats, departments, fees, and comparisons.</p>
        <form id="upload-form">
            <input type="text" id="college-name" placeholder="College name (optional)" />
            <input type="file" id="pdf-file" accept="application/pdf" />
            <button type="button" onclick="uploadPdf()">Upload & Index PDF</button>
        </form>
        <p id="upload-status"></p>
    </div>
    <div class="card">
        <h2>Ask a question</h2>
        <textarea id="question" rows="4" placeholder="Example: total computer science seats from all colleges in Trichy region"></textarea>
        <button type="button" onclick="sendQuestion()">Ask</button>
        <div class="result" id="answer"></div>
    </div>
    <script>
        async function uploadPdf() {
            const fileInput = document.getElementById('pdf-file');
            if (!fileInput.files.length) {
                document.getElementById('upload-status').innerText = 'Please choose a PDF file first.';
                return;
            }
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            formData.append('name', document.getElementById('college-name').value || '');
            document.getElementById('upload-status').innerText = 'Uploading...';
            const response = await fetch('/upload', { method: 'POST', body: formData });
            const result = await response.json();
            document.getElementById('upload-status').innerText = result.message || 'Upload failed.';
        }

        async function sendQuestion() {
            const text = document.getElementById('question').value.trim();
            if (!text) return;
            document.getElementById('answer').innerText = 'Searching...';
            const response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: text }),
            });
            const result = await response.json();
            document.getElementById('answer').innerText = result.answer || 'No answer available.';
        }
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(content=INDEX_HTML)


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...), name: str = Form("")):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    target_dir = Path(pdf_indexer.PDF_DIR)
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / file.filename
    with target_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    college_name = name.strip() or Path(file.filename).stem
    pdf_indexer.ingest_pdf(target_path, college_name)
    return JSONResponse({"message": f"Uploaded and indexed {college_name}."})


@app.get("/colleges")
def list_colleges():
    colleges = pdf_indexer.fetch_colleges()
    return JSONResponse({"colleges": colleges})


@app.post("/chat")
async def chat(request: Request):
    payload = await request.json()
    query = payload.get("query", "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query text is required.")
    answer = pdf_indexer.answer_for_query(query)
    return JSONResponse({"answer": answer})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
