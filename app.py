
import streamlit as st
import pandas as pd
import fitz  # PyMuPDF
from parser import extract_contact_info
from matcher import evaluate_candidate

# Helper to read uploaded PDF directly from memory
def read_uploaded_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text() + "\n"
    return text

# --- PAGE SETUP ---
st.set_page_config(page_title="AI Resume Screener", page_icon="📄", layout="wide")
st.title("📄 AI-Based Resume Screening & Ranking System")
st.markdown("Upload candidate resumes and paste a Job Description to automatically rank applicants using **BERT Semantic Matching** and **NLP Skill Extraction**.")

# --- INPUT SECTION ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Job Description")
    jd_text = st.text_area(
        "Paste the Job Description here:",
        height=220,
        placeholder="e.g., Looking for an AI/ML Intern skilled in Python, NLP, spaCy, BERT, Scikit-Learn..."
    )

with col2:
    st.subheader("2. Upload Candidate Resumes (PDF)")
    uploaded_files = st.file_uploader(
        "Drag and drop one or more PDF resumes:",
        type=["pdf"],
        accept_multiple_files=True
    )

# --- SCREENING BUTTON & RESULTS ---
if st.button("🚀 Screen & Rank Candidates", type="primary"):
    if not jd_text.strip():
        st.warning("Please paste a Job Description first!")
    elif not uploaded_files:
        st.warning("Please upload at least one PDF resume!")
    else:
        with st.spinner("Analyzing resumes with BERT & spaCy..."):
            results = []
            for file in uploaded_files:
                resume_text = read_uploaded_pdf(file)
                contact = extract_contact_info(resume_text)
                eval_data = evaluate_candidate(resume_text, jd_text)
                
                results.append({
                    "Candidate File": file.name,
                    "Final Score (%)": eval_data["final_score"],
                    "BERT Context (%)": eval_data["bert_score"],
                    "Skill Match (%)": eval_data["skill_score"],
                    "TF-IDF Keyword (%)": eval_data["tfidf_score"],
                    "Email": contact["email"],
                    "Phone": contact["phone"],
                    "Matched Skills": ", ".join(eval_data["matched_skills"]) or "None",
                    "Missing Skills": ", ".join(eval_data["missing_skills"]) or "None"
                })
            
            # Sort candidates from highest Final Score to lowest
            df = pd.DataFrame(results).sort_values(by="Final Score (%)", ascending=False).reset_index(drop=True)
            df.index += 1  # Start rank at 1 instead of 0
            
            st.success(f"Screened {len(uploaded_files)} resume(s) successfully!")
            st.subheader("🏆 Candidate Ranking Leaderboard")
            st.dataframe(df, use_container_width=True)
            
            # Export CSV Report (covers Week 8 Report Generation!)
            csv = df.to_csv(index_label="Rank").encode("utf-8")
            st.download_button(
                label="📥 Download Screening Report (CSV)",
                data=csv,
                file_name="candidate_screening_report.csv",
                mime="text/csv"
            )