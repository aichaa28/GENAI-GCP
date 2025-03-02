"""This module evaluates the chatbot."""
import numpy as np
from Backend.config import TABLE_NAME
from Backend.retrieve import connect_db
from langchain_core.prompts import ChatPromptTemplate
from agents import generate_response , llm

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
    questions_answers = get_random_questions(n)
    gpt_scores = []

    for question, true_answer in questions_answers:
        predicted_answer = generate_response(question, None, "english")
        if not predicted_answer:
            continue

        evaluation_prompt = ChatPromptTemplate.from_template(
            """
            Evaluate the response on a scale from 0 to 10 for the following criteria:
            
            **Question:** {question}
            **Expected Answer:** {true_answer}
            **Generated Answer:** {predicted_answer}
            
            Criteria:
            1. Relevance
            2. Coherence
            3. Factual Accuracy
            4. Fluency
            5. Completeness
            6. Naturalness
            7. Context Appropriateness
            8. Originality
            9. Tone Adherence
            10. Comprehensibility
            11. Source Justification
            12. Level of Detail
            13. Bias Absence
            14. Medical Realism
            15. Patient Adaptability
            16. RAG Verification
            17. Consistency with Known Facts
            18. Ability to Identify Uncertainty
            19. Robustness to Input Errors
            20. Compliance with Instructions
            
            Provide scores separated by commas.
            """
        )
        evaluation_chain = evaluation_prompt | llm
        evaluation = evaluation_chain.invoke({
            "question": question,
            "true_answer": true_answer,
            "predicted_answer": predicted_answer
        }).content

        scores = [float(s) for s in evaluation.split(",") if s.replace(".", "").isdigit()]
        gpt_scores.append(scores)

    criteria = [
        "Relevance", "Coherence", "Factual Accuracy", "Fluency", "Completeness", "Naturalness", 
        "Context Appropriateness", "Originality", "Tone Adherence", "Comprehensibility", "Source Justification", 
        "Level of Detail", "Bias Absence", "Medical Realism", "Patient Adaptability", "RAG Verification", 
        "Consistency with Known Facts", "Ability to Identify Uncertainty", "Robustness to Input Errors", 
        "Compliance with Instructions"
    ]
    mean_scores = np.mean(gpt_scores, axis=0).tolist() if gpt_scores else []
    final_scores = dict(zip(criteria, mean_scores))

    for crit, score in final_scores.items():
        print(f"{crit}: {score:.2f}/10")

    return final_scores


if __name__ == "__main__":
    evaluate_chatbot(2)
