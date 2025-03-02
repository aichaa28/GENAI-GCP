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
            dbname=DB_NAME.encode("utf-8").decode("utf-8"),
            user=DB_USER.encode("utf-8").decode("utf-8"),
            password=DB_PASSWORD.encode("utf-8").decode("utf-8"),
            host=DB_HOST.encode("utf-8").decode("utf-8"),
            port=DB_PORT.encode("utf-8").decode("utf-8")
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
                f"""SELECT answer, source, focus_area,
                embedding FROM {TABLE_NAME} WHERE embedding IS NOT NULL""")
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

    if similarities[best_idx] < 0.5:
        return None

    return {
        "answer": rows[best_idx][0],
        "source": rows[best_idx][1],
        "focus_area": rows[best_idx][2],
        "similarity": round(similarities[best_idx], 4)
    }


@lru_cache(maxsize=1000)
def get_all_embeddings_medoc() -> List[Tuple[str, str, str, str, List[float]]]:
    """Récupère les embeddings de PostgreSQL en excluant l'ID pour optimiser les requêtes."""
    conn = connect_db()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT drug, indication, side_effects, drug_interaction,
                embedding FROM ae_med_table WHERE embedding IS NOT NULL"""
            )
            rows = cur.fetchall()

        cleaned_rows = []
        for row in rows:
            try:
                embedding = json.loads(row[4]) if row[4] is not None else []
                cleaned_rows.append(
                    (row[0], row[1], row[2], row[3], embedding))
            except json.JSONDecodeError:
                print(f"Erreur de décodage JSON pour l'entrée : {row[0]}")

        return cleaned_rows
    finally:
        conn.close()


def find_best_matches_medoc(query_embedding: List[float], top_n: int = 3) -> Optional[dict]:
    """Retourne une moyenne des similarités des N meilleurs résultats."""
    rows = get_all_embeddings()
    if not rows:
        return None

    docs = [row[4] for row in rows]
    similarities = cosine_similarity([query_embedding], docs)[0]

    # Sélection des N meilleures similarités
    top_indices = np.argsort(similarities)[-top_n:]
    avg_similarity = np.mean([similarities[i] for i in top_indices])

    return {
        "top_n_avg_similarity": round(avg_similarity, 4),
        "top_matches": [
            {
                "drug": rows[i][0],
                "indication": rows[i][1],
                "side_effects": rows[i][2],
                "drug_interaction": rows[i][3],
                "similarity": round(similarities[i], 4)
            }
            for i in top_indices
        ]
    }
