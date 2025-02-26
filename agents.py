"""
This module contains utility functions 
for evaluating and logging metrics for the Gemini AI model.
"""
import os
import csv
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from config import API_KEY

# Load SentenceTransformer model
embedding_model = SentenceTransformer(
    "sentence-transformers/all-mpnet-base-v2")

# Initialize Gemini
# Initialize Gemini with API Key
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro", temperature=0.5, google_api_key=API_KEY)


def generate_embedding(text: str) -> List[float]:
    """Generate an embedding vector for the given text."""
    return embedding_model.encode(text, normalize_embeddings=True).tolist()


def evaluate_metrics(query: str, answer: str, generated_answer: str) -> Dict[str, Dict[str, float]]:
    """Evaluate similarity and quality metrics for the generated response."""
    query_embedding = generate_embedding(query)
    answer_embedding = generate_embedding(answer)
    generated_answer_embedding = generate_embedding(generated_answer)

    cosine_sim_answer = cosine_similarity(
        [query_embedding], [answer_embedding])[0][0]
    cosine_sim_generated_answer = cosine_similarity(
        [query_embedding], [generated_answer_embedding])[0][0]

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"])
    rouge_scores = scorer.score(answer, generated_answer)

    return {
        "cosine_similarity": {
            "answer": round(cosine_sim_answer, 4),
            "generated_answer": round(cosine_sim_generated_answer, 4)
        },
        "sts_similarity": round(cosine_sim_generated_answer, 4),
        "rouge_scores": {
            "rouge1": round(rouge_scores["rouge1"].fmeasure, 4),
            "rouge2": round(rouge_scores["rouge2"].fmeasure, 4),
            "rougeL": round(rouge_scores["rougeL"].fmeasure, 4)
        }
    }


def generate_response(question: str, context: str, language: str) -> str:
    """Generate an enriched response using Gemini AI model."""
    prompt_template = ChatPromptTemplate.from_template("""
    You are a medical AI assistant with expertise in clinical studies.
    Your goal is to provide accurate and structured answers.

    **Instructions :**
    - Prioritize medically validated information.
    - If the context is unclear, clarify before answering.
    - Use clear, professional language.
    - Cite sources if available.

    **Question:** {question}
    **Context:** {context}
    **Language:** {language}
    """)

    chain = prompt_template | llm
    response = chain.invoke({
        "question": question,
        "context": context,
        "language": language
    })
    return response.content


def log_metrics_to_csv(query: str, best_match: Dict[str, str] | None, response: str,
                       metrics: Dict[str, Dict[str, float]], response_time: float) -> None:
    """Log the query, response, and evaluation metrics to a CSV file."""
    file_path = "metrics_log.csv"
    file_exists = os.path.isfile(file_path)

    with open(file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow([
                "query", "best_match", "response", "cosine_similarity", "rouge1", "rouge2", "rougeL", "response_time"
            ])
        writer.writerow([
            query,
            best_match["answer"] if best_match else "None",
            response,
            metrics["cosine_similarity"].get("generated_answer", "N/A"),
            metrics["rouge_scores"].get("rouge1", "N/A"),
            metrics["rouge_scores"].get("rouge2", "N/A"),
            metrics["rouge_scores"].get("rougeL", "N/A"),
            round(response_time, 4)
        ])
