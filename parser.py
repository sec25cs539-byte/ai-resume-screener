
import fitz  # PyMuPDF
import docx  # python-docx
import re
import spacy
from spacy.matcher import PhraseMatcher

nlp = spacy.load("en_core_web_sm")

# Multi-industry Skills Database (Tech, Finance/Accounting, HR, Marketing)
SKILLS_DB = [
    # Tech & AI/ML
    "python", "java", "c++", "sql", "machine learning", "deep learning",
    "nlp", "natural language processing", "tensorflow", "pytorch", "keras",
    "scikit-learn", "pandas", "numpy", "spacy", "nltk", "bert", "transformers",
    "html", "css", "javascript", "react", "flask", "django", "fastapi",
    "git", "github", "docker", "aws", "data analysis", "computer vision",
    "opencv", "streamlit", "rest api", "mongodb", "postgresql",
    # Finance & Accounting
    "accounting", "bookkeeping", "financial analysis", "taxation", "gst",
    "auditing", "tally", "quickbooks", "sap", "financial modeling",
    "accounts payable", "accounts receivable", "payroll", "budgeting",
    "balance sheet", "reconciliation", "gaap", "ifrs", "excel", "power bi", "tableau",
    # Business, HR & Marketing
    "project management", "agile", "scrum", "recruitment", "talent acquisition",
    "onboarding", "employee relations", "hris", "seo", "sem", "content marketing",
    "social media marketing", "google analytics", "crm", "salesforce", "communication"
]

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return "\n".join([page.get_text() for page in doc])

def extract_text_from_docx(docx_path):
    doc = docx.Document(docx_path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_contact_info(text):
    email = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)
    phone = re.findall(r'\+?\d[\d -]{8,12}\d', text)
    return {
        "email": email[0] if email else "Not found",
        "phone": phone[0] if phone else "Not found"
    }

def extract_skills(text):
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(skill) for skill in SKILLS_DB]
    matcher.add("SKILLS", patterns)
    
    doc = nlp(text)
    matches = matcher(doc)
    
    found_skills = {doc[start:end].text.lower() for _, start, end in matches}
    return sorted(list(found_skills))