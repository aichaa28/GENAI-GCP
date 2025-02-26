""" this file contains the FastAPI code for the API endpoints.
    The API has two endpoints: get_sources and answer.
    The answer endpoint also evaluates the quality
    of the generated answer using various metrics """
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agents import (
    generate_embedding,
    evaluate_metrics,
    log_metrics_to_csv,
    generate_response
)
from retrieve import find_best_match


# Initialize FastAPI
app = FastAPI()


# Model for API requests
class QueryRequest(BaseModel):
    question: str
    temperature: float = 0.7
    language: str = "english"


# Endpoint to get sources
@app.post("/get_sources")
def get_sources(request: QueryRequest):
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
