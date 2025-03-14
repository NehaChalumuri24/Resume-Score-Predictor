import os
import fitz  # PyMuPDF for reading PDFs
import re
import joblib
import numpy as np
import streamlit as st
from nltk.tokenize import word_tokenize

# Ensure 'python-docx' is installed
try:
    from docx import Document
except ImportError:
    os.system("pip install python-docx")
    from docx import Document

# Load trained models
model = joblib.load("resume_score_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
scaler = joblib.load("scaler.pkl")

salary_model = joblib.load("salary_model.pkl")
salary_vectorizer = joblib.load("salary_vectorizer.pkl")
salary_scaler = joblib.load("salary_scaler.pkl")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = " ".join([page.get_text("text") for page in doc])
    return text

# Function to extract text from DOCX
def extract_text_from_docx(docx_path):
    doc = Document(docx_path)
    text = " ".join([para.text for para in doc.paragraphs])
    return text

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove special characters
    tokens = word_tokenize(text)
    return " ".join(tokens)

# Function to extract structured details from text
def extract_resume_details(text):
    exp_match = re.search(r'(\d{1,2})\s*(?:years?|yrs?)\s*(?:of experience|exp)?', text, re.IGNORECASE)
    experience = int(exp_match.group(1)) if exp_match else 0

    project_match = re.search(r'(\d+)\s*(?:projects?|project experience)', text, re.IGNORECASE)
    projects = int(project_match.group(1)) if project_match else 1

    return experience, projects

# Function to predict resume score
def predict_resume_score(text, experience, projects):
    text_vectorized = vectorizer.transform([text]).toarray()
    numeric_features = np.array([[experience, projects]])
    numeric_features_scaled = scaler.transform(numeric_features)
    X_final = np.hstack((text_vectorized, numeric_features_scaled))
    score = model.predict(X_final)[0]
    return round(score, 2)

# Function to predict salary in INR based on experience and projects
def predict_salary(text, experience, projects):
    text_vectorized = salary_vectorizer.transform([text]).toarray()
    numeric_features = np.array([[experience, projects]])
    numeric_features_scaled = salary_scaler.transform(numeric_features)
    X_final = np.hstack((text_vectorized, numeric_features_scaled))
    salary_usd = salary_model.predict(X_final)[0]
    salary_inr = salary_usd * 83  # Convert USD to INR (approx exchange rate)
    return round(salary_inr, 2)

# Streamlit UI
st.title("📄 AI Resume Scorer & Salary Predictor")
st.write("Upload your resume (PDF or DOCX) to get a score, salary prediction in INR, and skill improvement suggestions!")

uploaded_file = st.file_uploader("Upload Resume", type=["pdf", "docx"])

if uploaded_file is not None:
    if st.button("Submit"):
        file_extension = uploaded_file.name.split(".")[-1]
        file_path = os.path.join("temp." + file_extension)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Process the uploaded resume
        if file_extension == "pdf":
            resume_text = extract_text_from_pdf(file_path)
        elif file_extension == "docx":
            resume_text = extract_text_from_docx(file_path)
        else:
            st.error("Unsupported file format. Please upload a PDF or DOCX file.")
            st.stop()

        cleaned_text = preprocess_text(resume_text)
        experience, projects = extract_resume_details(cleaned_text)

        # Predict resume score
        score = predict_resume_score(cleaned_text, experience, projects)

        # Predict salary in INR
        salary = predict_salary(cleaned_text, experience, projects)

        # Suggest skills to improve
        job_skills = ["python", "machine learning", "deep learning", "nlp", "data analysis"]
        resume_words = set(word_tokenize(cleaned_text))
        skills_to_improve = [skill for skill in job_skills if skill.lower() not in resume_words]

        st.subheader("🏆 Resume Score:")
        st.write(f"**{score}/100**")

        st.subheader("💰 Predicted Salary Expectation (INR) Per Annum:")
        st.write(f"**₹{salary}**")

        st.subheader("📊 Extracted Details:")
        st.write(f"- **Experience:** {experience} years")
        st.write(f"- **Projects Count:** {projects}")

        st.subheader("💡 Skills to Improve:")
        if skills_to_improve:
            st.write(", ".join(skills_to_improve))
        else:
            st.write("✅ Your skills match well!")
