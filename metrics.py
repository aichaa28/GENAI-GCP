import os
import csv
import json
import numpy as np
from typing import Dict
from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu
from agents import generate_embedding


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


def evaluate_metrics_medoc(question: str, best_match: dict, generated_response: str) -> Dict[str, float]:    
    metrics = {}

    if not best_match or "drug" not in best_match:
        return {"error": "No valid medication match found"}

    #BLEU Score (similarité textuelle entre la question et le nom du médicament)
    reference = [question.lower().split()]
    candidate = best_match["drug"].lower().split()
    bleu_score = sentence_bleu(reference, candidate)
    metrics["BLEU"] = round(bleu_score, 4)

    #Similarité Cosine entre la question et le médicament trouvé
    try:
        question_embedding = generate_embedding(question)
        drug_embedding = generate_embedding(best_match["drug"])
        similarity = cosine_similarity([question_embedding], [drug_embedding])[0][0]
        metrics["cosine_similarity"] = round(similarity, 4)
    except Exception as e:
        print(f"Erreur lors du calcul de la similarité cosine : {e}")
        metrics["cosine_similarity"] = 0.0  # Valeur par défaut en cas d'erreur

    return metrics


