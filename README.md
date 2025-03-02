# GENAI-GCP: AI-Powered Medical Chatbot

## 📌 Project Overview
This project aims to develop an AI-powered medical chatbot capable of assisting users with general health-related inquiries. The chatbot leverages **Natural Language Processing (NLP)** and **multimodal AI (Gemini 1.5)** to analyze **medical images** and **text-based queries**.

---
## 🚀 Features

### 1️⃣ **Medical Chatbot**
- Provides responses to general medical inquiries.
- Uses **Google Gemini 1.5** and **Hugging Face models** for AI-based reasoning.

### 2️⃣ **Medical Image Analysis**
- Supports **medical image uploads** (X-rays, MRIs, lesions, etc.).
- Uses **AI-powered anomaly detection**.
- Offers **comparative analysis** over multiple images.

### 3️⃣ **Medicine Identification**
- Recognizes **pills, tablets, and drug packaging**.
- Extracts **relevant medical information**.
- Matches against a **dataset of known medications**.

### 4️⃣ **Cloud & Deployment**
- Deployed on **Google Cloud Platform (GCP)**.

### 5️⃣ **Real-Time Metrics & Logging**
- Logs **response times, similarity scores, and accuracy**.
- Provides **feedback collection** from users.


---
## 🏗 Project Structure
```
GENAI-GCP/
│── backend/
│   ├── agents.py          # Manages AI agents & Gemini interactions
│   ├── api.py             # FastAPI backend to process queries
│   ├── eval2.py          # LLM evoluation of the model
│   ├── config.py          # Configuration & API keys
│   ├── ingest.py          # Loads and preprocesses dataset
│   └── retrieve.py        # Fetches embeddings & best matches
│
│── frontend/
│   ├── app.py             # Streamlit frontend chatbot UI
│
│── graph/
│   ├── graph.py           # Generates analytics & visualizations
│   └── Images             # Some images of our graphs
│
│── dataset/
│   ├── dataset.csv        # Medical dataset for reference
│   ├── medquad.csv        # Medical QA dataset for model training
│   ├── medoc_info.csv     # Medecine QA dataset for model training
│   └── feedback.csv       # Logs user feedback
│
│── evaluation/
│   ├── eval.py            # Evaluation metrics for chatbot responses
│   └── metrics_log.csv    # Tracks chatbot response metrics
│
├── Dockerfile         # Dockerfile for building the frontend container
│── Dockerfile_api     # Dockerfile for building the backend container
│
│── requirements.txt       # Python dependencies
│── README.md              # Project documentation (this file)

```
## Model Card

![Model Card](Model_card)
---
## 🔧 Installation & Setup

### **1️⃣ Clone the repository**
```bash
git clone https://github.com/aichaa28/GENAI-GCP.git
cd GENAI-GCP
```

### **2️⃣ Install dependencies**
```bash
pip install -r requirements.txt
```

### **3️⃣ Set up environment variables**
- Configure API keys for **Google Gemini**, **Hugging Face**, **PostgreSQL**, etc.

### **4️⃣ Run the application**
```bash
streamlit run app.py  # Start the chatbot UI
uvicorn api:app --reload  # Start the FastAPI backend
```

---
## 📊 Data & Model Usage
- Uses **Gemini 1.5** and **Hugging Face models** for **multimodal analysis (text & image)**.
- Stores **conversation logs** and **feedback** for model improvement.
- Cloud deployment using **Google Cloud Platform (GCP)**.

---
