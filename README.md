# College Admission Chatbot

A lightweight local chatbot for college admission queries. Colleges can upload PDF admission brochures, and students can ask questions about seats, departments, fees, comparisons, and totals.

## Features
- Upload college admission PDF documents
- Extract and index PDF text locally
- Answer student queries using PDF content
- Compare colleges and aggregate seat totals
- No external chat APIs required

## Requirements
- Python 3.10+
- 8GB RAM system

## Setup
1. Create a virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Run
```bash
python app.py
```

Then open http://127.0.0.1:8000 in your browser.

## How it works
- Uploaded PDFs are stored in `data/pdfs/` permanently.
- The service extracts text from each PDF and indexes it into a local SQLite database.
- Student questions are answered by searching the indexed text and returning relevant facts.

## Notes
- This is a prototype designed for quick setup and local use.
- For the best results, upload well-formatted college admission brochures.
- The system does not call any external chat APIs like ChatGPT/Gemini.
