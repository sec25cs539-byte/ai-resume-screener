
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Import our own functions from Lesson 2!
from parser import extract_text_from_pdf, extract_skills

# Load a fast, lightweight pre-trained BERT model (downloads once, ~80MB)
print("Loading BERT model...")
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

def get_tfidf_score(resume_text, jd_text):
    # Convert both texts into TF-IDF number vectors
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform([resume_text, jd_text])
    
    # Calculate Cosine Similarity between vector 0 (resume) and vector 1 (JD)
    similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    return round(similarity * 100, 2)

def get_bert_score(resume_text, jd_text):
    # Convert both texts into semantic BERT embeddings
    embeddings = bert_model.encode([resume_text, jd_text])
    
    # Calculate Cosine Similarity
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return round(float(similarity) * 100, 2)

def evaluate_candidate(resume_text, jd_text):
    # 1. Get AI similarity scores
    tfidf_score = get_tfidf_score(resume_text, jd_text)
    bert_score = get_bert_score(resume_text, jd_text)
    
    # 2. Compare Skills between Resume and Job Description
    resume_skills = set(extract_skills(resume_text))
    jd_skills = set(extract_skills(jd_text))
    
    matched_skills = sorted(list(resume_skills.intersection(jd_skills)))
    missing_skills = sorted(list(jd_skills - resume_skills))
    
    # Calculate Skill Match %
    if len(jd_skills) > 0:
        skill_score = round((len(matched_skills) / len(jd_skills)) * 100, 2)
    else:
        skill_score = 0.0
        
    # 3. Hybrid Final Score (60% BERT Context + 40% Exact Skill Match)
    final_score = round((0.6 * bert_score) + (0.4 * skill_score), 2)
    
    return {
        "tfidf_score": tfidf_score,
        "bert_score": bert_score,
        "skill_score": skill_score,
        "final_score": final_score,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills
    }

# --- TEST OUR AI MATCHER ---
if __name__ == "__main__":
    # Sample Job Description
    sample_jd = """
    We are looking for an AI/ML Intern proficient in Python, Machine Learning, 
    NLP, spaCy, and BERT. Experience with Scikit-Learn, Pandas, and building 
    web apps using Streamlit or FastAPI is a strong plus.
    """
    
    resume_text = extract_text_from_pdf("sample_resume.pdf")
    result = evaluate_candidate(resume_text, sample_jd)
    
    print("\n=== AI SCREENING REPORT ===")
    print(f"TF-IDF Score (Keyword) : {result['tfidf_score']}%")
    print(f"BERT Score (Semantic)  : {result['bert_score']}%")
    print(f"Skill Match Score      : {result['skill_score']}%")
    print(f"FINAL RANKING SCORE    : {result['final_score']}%")
    print("Matched Skills         :", result['matched_skills'])
    print("Missing Skills         :", result['missing_skills'])