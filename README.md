# **GENAI-GCP: AI-Powered Medical Chatbot**
## 📌 Project Overview
This project aims to develop an AI-powered medical chatbot capable of assisting users with general health-related inquiries. The chatbot leverages Natural Language Processing (NLP) and multimodal AI (Gemini 1.5) to analyze medical images and text-based queries.

## 🚀 Features
### 1️⃣ Medical Chatbot
Provides responses to general medical inquiries.
Supports multiple languages.
Uses Google Gemini 1.5 and Hugging Face models for AI-based reasoning.
### 2️⃣ Medical Image Analysis
Supports medical image uploads (X-rays, MRIs, lesions, etc.).
Uses AI-powered anomaly detection.
Offers comparative analysis over multiple images.
### 3️⃣ Medicine Identification
Recognizes pills, tablets, and drug packaging.
Extracts relevant medical information.
Matches against a dataset of known medications.
### 4️⃣ Cloud & Deployment
Deployed on Google Cloud Platform (GCP).
Uses Vertex AI for scalable AI model serving.
Supports LangFuse for API monitoring and performance tracking.
### 5️⃣ Real-Time Metrics & Logging
Logs response times, similarity scores, and accuracy.
Uses LangFuse for monitoring API calls.
Provides feedback collection from users.
### 6️⃣ Fine-Tuned Medical AI
Integrates custom Hugging Face models for medical analysis.
Fine-tunes responses for specialized domains.
Uses cloud-based model hosting for real-time inference.
## 🏗 Project Structure
GENAI-GCP/
│── agents.py          # Manages AI agents & Gemini interactions
│── api.py             # FastAPI backend to process queries
│── app.py             # Streamlit frontend chatbot UI
│── config.py          # Configuration & API keys
│── dataset.csv        # Medical dataset for reference
│── eval.py            # Evaluation metrics for chatbot responses
│── feedback.csv       # Logs user feedback
│── graph.py           # Generates analytics & visualizations
│── ingest.py          # Loads and preprocesses dataset
│── medquad.csv        # Medical QA dataset for model training
│── metrics_log.csv    # Tracks chatbot response metrics
│── retrieve.py        # Fetches embeddings & best matches
│── requirements.txt   # Python dependencies
│── test_connection.py # Database connectivity test
│── README.md          # Project documentation (this file)
## 🔧 Installation & Setup
### 1️⃣ Clone the repository
git clone https://github.com/aichaa28/GENAI-GCP.git
cd GENAI-GCP
### 2️⃣ Install dependencies
pip install -r requirements.txt
### 3️⃣ Set up environment variables
Configure API keys for Google Gemini, Hugging Face, PostgreSQL, etc., in config.py.
### 4️⃣ Run the application
streamlit run app.py  # Start the chatbot UI
uvicorn api:app --reload  # Start the FastAPI backend
## 📊 Data & Model Usage
Uses Gemini 1.5 and Hugging Face models for multimodal analysis (text & image).
Stores conversation logs and feedback for model improvement.
Supports fine-tuning with medquad.csv & dataset.csv.
Cloud deployment using Google Cloud Platform (GCP).
