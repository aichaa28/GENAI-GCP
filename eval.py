"""This module evaluate the chatbot."""
import requests
import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity
from config import TABLE_NAME
from retrieve import connect_db
from agents import generate_embedding


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

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])

    for question, true_answer in questions_answers:
        response = requests.post(
            "http://127.0.0.1:8000/answer", json={"question": question}, timeout=10)

        if response.status_code == 200:
            data = response.json()
            predicted_answer = data.get("answer", "")

            # BLEU Score
            bleu = sentence_bleu([true_answer.split()],
                                 predicted_answer.split())
            bleu_scores.append(bleu)

            # F1 Score (token-matching)
            true_tokens, pred_tokens = set(
                true_answer.split()), set(predicted_answer.split())
            common_tokens = true_tokens.intersection(pred_tokens)
            precision = len(common_tokens) / \
                len(pred_tokens) if pred_tokens else 0
            recall = len(common_tokens) / \
                len(true_tokens) if true_tokens else 0
            f1 = 2 * (precision * recall) / (precision +
                                             recall) if (precision + recall) else 0
            f1_scores.append(f1)

            # Cosine Similarity
            true_embedding = generate_embedding(true_answer)
            predicted_embedding = generate_embedding(predicted_answer)
            cosine_sim = cosine_similarity(
                [true_embedding], [predicted_embedding])[0][0]
            cosine_similarities.append(cosine_sim)

            # ROUGE Score
            rouge_scores = scorer.score(true_answer, predicted_answer)
            rouge_scores_list.append({
                "rouge1": round(rouge_scores["rouge1"].fmeasure, 4),
                "rouge2": round(rouge_scores["rouge2"].fmeasure, 4),
                "rougeL": round(rouge_scores["rougeL"].fmeasure, 4)
            })

    # Affichage des résultats
    print("\n--- Évaluation du Chatbot ---")
    print(f"BLEU Score Moyen: {np.mean(bleu_scores):.4f}")
    print(f"F1 Score Moyen: {np.mean(f1_scores):.4f}")
    print(f"Cosine Similarity Moyenne: {np.mean(cosine_similarities):.4f}")
    rouge_avg = {
        "rouge1": np.mean([r["rouge1"] for r in rouge_scores_list]),
        "rouge2": np.mean([r["rouge2"] for r in rouge_scores_list]),
        "rougeL": np.mean([r["rougeL"] for r in rouge_scores_list])
    }
    print(f"ROUGE Scores Moyens: {rouge_avg}")

    return {
        "bleu": np.mean(bleu_scores),
        "f1": np.mean(f1_scores),
        "cosine_similarity": np.mean(cosine_similarities),
        "rouge": rouge_avg
    }


if __name__ == "__main__":
    evaluate_chatbot(10)
