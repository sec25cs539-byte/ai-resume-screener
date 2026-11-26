
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from matcher import evaluate_candidate

# Sample benchmark dataset of (Resume Text, Job Description, Human Ground Truth: 1=Good Match, 0=Poor Match)
test_cases = [
    (
        "Experienced Python Developer skilled in Machine Learning, NLP, spaCy, BERT, and Scikit-Learn.",
        "Hiring AI/ML Engineer with Python, NLP, BERT, and Scikit-Learn experience.",
        1
    ),
    (
        "Senior Accountant with 5 years in Tally, GST, Auditing, Balance Sheet Reconciliation, and Excel.",
        "Looking for an Accountant proficient in Auditing, GST, Tally, and Financial Modeling.",
        1
    ),
    (
        "Graphic Designer expert in Adobe Photoshop, Illustrator, and UI/UX wireframing.",
        "Hiring AI/ML Engineer with Python, NLP, BERT, and Scikit-Learn experience.",
        0
    ),
    (
        "Python Backend Developer experienced in Django, PostgreSQL, Docker, and AWS.",
        "Looking for an Accountant proficient in Auditing, GST, Tally, and Financial Modeling.",
        0
    ),
    (
        "Data Analyst skilled in SQL, Python, Pandas, Power BI, and Excel reporting.",
        "Seeking Data Analyst with strong SQL, Python, Pandas, and Power BI skills.",
        1
    ),
    (
        "HR Generalist handling Recruitment, Onboarding, Payroll, and Employee Relations.",
        "Seeking Data Analyst with strong SQL, Python, Pandas, and Power BI skills.",
        0
    )
]

# Set a threshold: if Final Score >= 45%, model predicts "Good Match" (1), else "Poor Match" (0)
THRESHOLD = 45.0

y_true = []
y_pred = []

print("=== RUNNING MODEL BENCHMARK ===")
for i, (resume, jd, actual_label) in enumerate(test_cases, 1):
    result = evaluate_candidate(resume, jd)
    predicted_label = 1 if result["final_score"] >= THRESHOLD else 0
    
    y_true.append(actual_label)
    y_pred.append(predicted_label)
    print(f"Case {i}: Score = {result['final_score']}% | Predicted = {predicted_label} | Actual = {actual_label}")

print("\n=== FINAL EVALUATION METRICS ===")
print(f"Precision : {precision_score(y_true, y_pred) * 100:.2f}%")
print(f"Recall    : {recall_score(y_true, y_pred) * 100:.2f}%")
print(f"F1-Score  : {f1_score(y_true, y_pred) * 100:.2f}%")