import os
import re
import joblib
import numpy as np
import streamlit as st
import subprocess
import sys

# Ensure required libraries are installed
try:
    import fitz  # PyMuPDF
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pymupdf"])
    import fitz

try:
    import nltk
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "nltk"])
    import nltk

# Ensure NLTK resources are downloaded before use
def ensure_nltk_resources():
    resources = ["punkt", "stopwords", "wordnet"]
    for resource in resources:
        try:
            if resource == "punkt":
                nltk.data.find("tokenizers/punkt")
            else:
                nltk.data.find(f"corpora/{resource}")
        except LookupError:
            nltk.download(resource)

ensure_nltk_resources()

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

try:
    from docx import Document
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "python-docx"])
    from docx import Document

# Load models safely
def load_model(file_name):
    if os.path.exists(file_name):
        return joblib.load(file_name)
    else:
        st.error(f"❌ Missing model file: {file_name}. Ensure it's uploaded in your project directory.")
        st.stop()

model = load_model("resume_score_model.pkl")
vectorizer = load_model("vectorizer.pkl")
scaler = load_model("scaler.pkl")
salary_model = load_model("salary_model.pkl")
salary_vectorizer = load_model("salary_vectorizer.pkl")
salary_scaler = load_model("salary_scaler.pkl")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    try:
        with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
            text = " ".join([page.get_text("text") for page in doc])
        return text
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

# Function to extract text from DOCX
def extract_text_from_docx(docx_file):
    try:
        doc = Document(docx_file)
        text = " ".join([para.text for para in doc.paragraphs])
        return text
    except Exception as e:
        return f"Error reading DOCX: {str(e)}"

# Preprocessing text function
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove special characters
    tokens = word_tokenize(text)  # Tokenize words
    return " ".join(tokens)  # Return cleaned text

# Extract resume details (experience & projects)
def extract_resume_details(text):
    exp_match = re.search(r'(\d{1,2})\s*(?:years?|yrs?)\s*(?:of experience|exp)?', text, re.IGNORECASE)
    experience = int(exp_match.group(1)) if exp_match else 0

    project_match = re.search(r'(\d+)\s*(?:projects?|project experience)', text, re.IGNORECASE)
    projects = int(project_match.group(1)) if project_match else 1

    return experience, projects

# Predict resume score
def predict_resume_score(text, experience, projects):
    try:
        text_vectorized = vectorizer.transform([text]).toarray()
        numeric_features = np.array([[experience, projects]])
        numeric_features_scaled = scaler.transform(numeric_features)
        X_final = np.hstack((text_vectorized, numeric_features_scaled))
        score = model.predict(X_final)[0]
        return round(score, 2)
    except Exception as e:
        st.error(f"Error in resume scoring: {str(e)}")
        return None

# Predict salary expectation
def predict_salary(text, experience, projects):
    try:
        text_vectorized = salary_vectorizer.transform([text]).toarray()
        numeric_features = np.array([[experience, projects]])
        numeric_features_scaled = salary_scaler.transform(numeric_features)
        X_final = np.hstack((text_vectorized, numeric_features_scaled))
        salary_usd = salary_model.predict(X_final)[0]
        salary_inr = salary_usd * 83  # Convert to INR
        return round(salary_inr, 2)
    except Exception as e:
        st.error(f"Error in salary prediction: {str(e)}")
        return None

# Streamlit UI
st.title("📄 AI Resume Scorer & Salary Predictor")
st.write("Upload your resume (PDF or DOCX) to get a score, salary prediction in INR, and skill improvement suggestions!")

uploaded_file = st.file_uploader("Upload Resume", type=["pdf", "docx"])

if uploaded_file is not None:
    if st.button("Submit"):
        # Extract text based on file type
        file_extension = uploaded_file.name.split(".")[-1]

        if file_extension == "pdf":
            resume_text = extract_text_from_pdf(uploaded_file)
        elif file_extension == "docx":
            resume_text = extract_text_from_docx(uploaded_file)
        else:
            st.error("❌ Unsupported file format. Please upload a PDF or DOCX file.")
            st.stop()

        # Check if resume text is extracted correctly
        if "Error reading" in resume_text:
            st.error(resume_text)
            st.stop()

        cleaned_text = preprocess_text(resume_text)
        experience, projects = extract_resume_details(cleaned_text)

        # Predict score and salary
        score = predict_resume_score(cleaned_text, experience, projects)
        salary = predict_salary(cleaned_text, experience, projects)

        # Suggest missing skills
        job_skills = ["python", "machine learning", "deep learning", "nlp", "data analysis"]
        resume_words = set(word_tokenize(cleaned_text))
        skills_to_improve = [skill for skill in job_skills if skill.lower() not in resume_words]

        # Display results
        st.subheader("🏆 Resume Score:")
        st.write(f"*{score}/100*") if score is not None else st.write("Error in scoring.")

        st.subheader("💰 Predicted Salary Expectation (INR) Per Annum:")
        st.write(f"*₹{salary}*") if salary is not None else st.write("Error in salary prediction.")

        st.subheader("📊 Extracted Details:")
        st.write(f"- *Experience:* {experience} years")
        st.write(f"- *Projects Count:* {projects}")

        st.subheader("💡 Skills to Improve:")
        if skills_to_improve:
            st.write(", ".join(skills_to_improve))
        else:
            st.write("✅ Your skills match well!")
