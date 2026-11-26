
from fastapi import FastAPI, UploadFile, File, Form
import fitz
from parser import extract_contact_info
from matcher import evaluate_candidate

app = FastAPI(
    title="AI Resume Screening API",
    description="REST API for parsing resumes and ranking candidates using BERT & NLP.",
    version="1.0.0"
)

@app.get("/")
def health_check():
    return {"status": "online", "message": "AI Resume Screening API is running"}

@app.post("/screen-resume/")
async def screen_resume(
    job_description: str = Form(...),
    resume_file: UploadFile = File(...)
):
    # Read uploaded PDF
    pdf_bytes = await resume_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    resume_text = "\n".join([page.get_text() for page in doc])
    
    # Extract info & run AI evaluation
    contact = extract_contact_info(resume_text)
    scores = evaluate_candidate(resume_text, job_description)
    
    return {
        "filename": resume_file.filename,
        "candidate_contact": contact,
        "evaluation": scores
    }