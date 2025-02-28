"""This module evaluates the chatbot."""
import requests
import numpy as np
from config import TABLE_NAME
from retrieve import connect_db
from metrics import evaluate_metrics


def get_random_questions(n):
    """Récupère n questions aléatoires depuis la base de données."""
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute(
        f"SELECT question, answer FROM {TABLE_NAME} ORDER BY RANDOM() LIMIT %s", (n,))
    data = cursor.fetchall()
    cursor.close()
    conn.close()
    return data


def evaluate_chatbot(n):
    """Teste le chatbot sur n questions et calcule les métriques."""
    questions_answers = get_random_questions(n)
    bleu_scores, f1_scores, cosine_similarities, rouge_scores_list = [], [], [], []

    for question, true_answer in questions_answers:
        response = requests.post(
            "http://127.0.0.1:8000/answer", json={"question": question}, timeout=500
        )

        if response.status_code == 200:
            data = response.json()
            predicted_answer = data.get("answer", "")

            # Évaluation avec evaluate_metrics()
            metrics = evaluate_metrics(question, true_answer, predicted_answer)

            # BLEU Score (utilisé ici comme STS Similarity)
            bleu_scores.append(metrics["sts_similarity"])

            # F1 Score (token-matching)
            true_tokens, pred_tokens = set(true_answer.split()), set(predicted_answer.split())
            common_tokens = true_tokens.intersection(pred_tokens)
            precision = len(common_tokens) / len(pred_tokens) if pred_tokens else 0
            recall = len(common_tokens) / len(true_tokens) if true_tokens else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0
            f1_scores.append(f1)

            # Cosine Similarity
            cosine_similarities.append(metrics["cosine_similarity"]["generated_answer"])

            # ROUGE Scores
            rouge_scores_list.append(metrics["rouge_scores"])

    # Calcul des moyennes
    rouge_avg = {
        "rouge1": np.mean([r["rouge1"] for r in rouge_scores_list]),
        "rouge2": np.mean([r["rouge2"] for r in rouge_scores_list]),
        "rougeL": np.mean([r["rougeL"] for r in rouge_scores_list]),
    }

    # Affichage des résultats
    print("\n--- Évaluation du Chatbot ---")
    print(f"BLEU Score Moyen: {np.mean(bleu_scores):.4f}")
    print(f"F1 Score Moyen: {np.mean(f1_scores):.4f}")
    print(f"Cosine Similarity Moyenne: {np.mean(cosine_similarities):.4f}")
    print(f"ROUGE Scores Moyens: {rouge_avg}")

    return {
        "bleu": np.mean(bleu_scores),
        "f1": np.mean(f1_scores),
        "cosine_similarity": np.mean(cosine_similarities),
        "rouge": rouge_avg
    }


if __name__ == "__main__":
    evaluate_chatbot(30)
