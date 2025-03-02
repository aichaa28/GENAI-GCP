""" this file contains the FastAPI code for the API endpoints.
    The API has two endpoints: get_sources and answer.
    The answer endpoint also evaluates the quality
    of the generated answer using various metrics """
import time
import shutil
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from Backend.agents import (generate_embedding, generate_response,
                    extract_text_from_image, correct_medication_name,
                    get_medication_details)
from Evaluation.metrics import (evaluate_metrics, log_metrics_to_csv,
                                evaluate_metrics_medoc)
from retrieve import find_best_match, find_best_matches_medoc


# Initialize FastAPI
app = FastAPI()


# Model for API requests
class QueryRequest(BaseModel):
    """ Représente une requête pour poser une question
    à l'API. Cette classe contient 
la question, la température de la réponse, et la langue. """
    question: str
    temperature: float = 0.7
    language: str = "english"


# Endpoint to get sources
@app.post("/get_sources")
def get_sources(request: QueryRequest):
    """ Récupère les sources correspondant à la question envoyée. 
    Args: request
    (QueryRequest): La requête contenant la question, la température et la langue. 
    Returns: dict: Le meilleur match des sources. """
    start_time = time.time()
    query_embedding = generate_embedding(request.question)
    best_match = find_best_match(query_embedding)

    if best_match:
        response_time = time.time() - start_time
        print(f"Response time for get_sources: {response_time:.4f} seconds")
        return best_match

    response_time = time.time() - start_time
    print(f"Response time for get_sources: {response_time:.4f} seconds")
    raise HTTPException(status_code=404, detail="No relevant document found.")


# Endpoint to generate an enriched answer with Gemini
@app.post("/answer")
def answer(request: QueryRequest):
    start_time = time.time()
    query_embedding = generate_embedding(request.question)
    best_match = find_best_match(query_embedding)

    if not best_match:
        response_time = time.time() - start_time
        print(
            f"No match found: {response_time:.4f} s")
        return {"message": """I couldn't find relevant information.
                Answering based on general knowledge."""}

    response = generate_response(
        request.question, best_match['answer'], request.language)

    metrics = evaluate_metrics(
        request.question, best_match["answer"], response)
    print(f"Metrics: {metrics}")
    response_time = time.time() - start_time
    print(
        f"Response time for answer: {response_time:.4f} s")
    log_metrics_to_csv(request.question, best_match,
                       response, metrics, response_time)

    return {
        "answer": response,
        "source": best_match["source"],
        "focus_area": best_match["focus_area"],
        "similarity": best_match["similarity"],
        "metrics": metrics,
        "response_time": round(response_time, 4)
    }


# Endpoint pour rechercher un médicament
@app.post("/get_medication_info")
def get_medication_info(request: QueryRequest):
    start_time = time.time()
    query_embedding = generate_embedding(request.question)
    # Utilisation de la nouvelle fonction
    best_match = find_best_matches_medoc(query_embedding)

    if best_match:
        response_time = time.time() - start_time
        print(
            f"Response time for get_medication_info: {response_time:.4f} seconds")
        return {
            "drug": best_match["drug"],
            "indication": best_match["indication"],
            "side_effects": best_match["side_effects"],
            "drug_interaction": best_match["drug_interaction"],
            "similarity": best_match["similarity"],
            "response_time": round(response_time, 4)
        }

    response_time = time.time() - start_time
    print(
        f"Response time for get_medication_info: {response_time:.4f} seconds")
    raise HTTPException(
        status_code=404, detail="No relevant medication found.")

# Endpoint pour générer une réponse enrichie avec Gemini


@app.post("/answer_medication")
def answer_medication(request: QueryRequest):
    start_time = time.time()
    query_embedding = generate_embedding(request.question)
    best_match = find_best_matches_medoc(query_embedding)

    if not best_match:
        response_time = time.time() - start_time
        print(f"No match found: {response_time:.4f} s")
        return {"message": "I couldn't find relevant medication information. Answering based on general knowledge."}

    response = generate_response(
        request.question, best_match['drug'], request.language)

    metrics = evaluate_metrics_medoc(
        request.question, best_match["drug"], response)
    print(f"Metrics: {metrics}")
    response_time = time.time() - start_time
    print(f"Response time for answer_medication: {response_time:.4f} s")

    return {
        "answer": response,
        "drug": best_match["drug"],
        "indication": best_match["indication"],
        "side_effects": best_match["side_effects"],
        "drug_interaction": best_match["drug_interaction"],
        "similarity": best_match["similarity"],
        "metrics": metrics,
        "response_time": round(response_time, 4)
    }

# Endpoint pour traiter une image de médicament


@app.post("/process_medication_image")
async def process_medication(image: UploadFile = File(...)):
    """
    Reçoit une image de médicament, extrait et corrige le texte,
    puis retourne les détails du médicament.
    """
    # Sauvegarde temporaire de l'image
    temp_image_path = f"temp_{image.filename}"
    with open(temp_image_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    try:
        extracted_text = extract_text_from_image(temp_image_path)
        print(extracted_text)
        corrected_name = correct_medication_name(extracted_text)
        medication_info = get_medication_details(corrected_name, "English")

        # Nettoyage du fichier temporaire
        os.remove(temp_image_path)

        return {
            "status": "success",
            "corrected_name": corrected_name,
            "medication_info": medication_info
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}
