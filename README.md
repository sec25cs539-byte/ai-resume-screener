
# AI-Based Resume Screening & Ranking System

An end-to-end NLP & Machine Learning application that automates candidate resume parsing, skill extraction, and semantic ranking against job descriptions.

## Architecture & Tech Stack
- **Document Parsing:** PyMuPDF (`fitz`), `python-docx`, Regex
- **NLP & Information Extraction:** spaCy (`en_core_web_sm`, `PhraseMatcher`)
- **AI Matching & Ranking:** Scikit-Learn (TF-IDF Cosine Similarity) + HuggingFace Sentence-Transformers (`all-MiniLM-L6-v2` BERT Embeddings)
- **Web Dashboard:** Streamlit & Pandas
- **REST API:** FastAPI & Uvicorn (with Swagger UI at `/docs`)

## How to Run Locally
1. Activate virtual environment: `venv\Scripts\activate`
2. Install dependencies: `pip install -r requirements.txt`
3. Run Web Dashboard: `streamlit run app.py`
4. Run REST API: `uvicorn api:app --reload`
5. Run Model Evaluation: `python evaluate.py`