""" this module is responsible for retrieving embeddings
    from the database and finding the best match for a given query embedding.
    """
import json
from functools import lru_cache
from typing import List, Tuple, Optional
import psycopg2
import numpy as np
from fastapi import HTTPException
from sklearn.metrics.pairwise import cosine_similarity
from config import TABLE_NAME, DB_USER, DB_NAME, DB_HOST, DB_PORT, DB_PASSWORD


def connect_db():
    """Establish a secure connection to PostgreSQL."""
    try:
        return psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
    except psycopg2.Error as e:
        raise HTTPException(
            status_code=500, detail=f"DB connection error: {str(e)}") from e


@lru_cache(maxsize=1000)
def get_all_embeddings() -> List[Tuple[str, str, str, List[float]]]:
    """Retrieve cached embeddings from PostgreSQL to reduce query load."""
    conn = connect_db()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT answer, source, focus_area, embedding FROM {TABLE_NAME} WHERE embedding IS NOT NULL")
            rows = cur.fetchall()

        # Vérifier et convertir les embeddings valides uniquement
        cleaned_rows = []
        for row in rows:
            try:
                embedding = json.loads(row[3]) if row[3] is not None else []
                cleaned_rows.append((row[0], row[1], row[2], embedding))
            except json.JSONDecodeError:
                print(f"Erreur de décodage JSON pour l'entrée : {row[0]}")

        return cleaned_rows
    finally:
        conn.close()


def find_best_match(query_embedding: List[float]) -> Optional[dict]:
    """Find the best match for the given query embedding."""
    rows = get_all_embeddings()
    if not rows:
        return None

    docs = [row[3] for row in rows]
    similarities = cosine_similarity([query_embedding], docs)[0]
    best_idx = np.argmax(similarities)

    if similarities[best_idx] < 0.75:
        return None

    return {
        "answer": rows[best_idx][0],
        "source": rows[best_idx][1],
        "focus_area": rows[best_idx][2],
        "similarity": round(similarities[best_idx], 4)
    }
