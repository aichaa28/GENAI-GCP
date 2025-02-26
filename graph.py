"""This module handles graph-related operations such as visualization and analysis."""
import os
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# Chemin du fichier CSV et du dossier de sortie
FILE_PATH = "metrics_log.csv"
OUTPUT_DIR = "graphs"

# Création du dossier de sortie s'il n'existe pas
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    """Charge les données du fichier CSV."""
    try:
        return pd.read_csv(FILE_PATH)
    except FileNotFoundError:
        st.error("❌ Le fichier de métriques n'a pas été trouvé.")
        return None


def plot_cosine_similarity_evolution(df):
    """Génère et sauvegarde l'évolution de la similarité cosinus."""
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df["cosine_similarity"], marker='o',
             linestyle='-', label="Cosine Similarity")
    plt.xlabel("Requêtes")
    plt.ylabel("Cosine Similarity")
    plt.title("Évolution de la Similarité Cosinus")
    plt.legend()
    plt.grid(True)
    path = os.path.join(OUTPUT_DIR, "cosine_similarity.png")
    plt.savefig(path)
    plt.close()
    return path


def plot_rouge_scores(df):
    """Génère et sauvegarde l'évolution des scores ROUGE."""
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df["rouge1"], marker='o',
             linestyle='-', label="ROUGE-1")
    plt.plot(df.index, df["rouge2"], marker='s',
             linestyle='-', label="ROUGE-2")
    plt.plot(df.index, df["rougeL"], marker='^',
             linestyle='-', label="ROUGE-L")
    plt.xlabel("Requêtes")
    plt.ylabel("Score ROUGE")
    plt.title("Évolution des Scores ROUGE")
    plt.legend()
    plt.grid(True)
    path = os.path.join(OUTPUT_DIR, "rouge_scores.png")
    plt.savefig(path)
    plt.close()
    return path


def plot_response_time(df):
    """Génère et sauvegarde l'évolution du temps de réponse."""
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df["response_time"], marker='x',
             linestyle='-', color='red', label="Temps de Réponse")
    plt.xlabel("Requêtes")
    plt.ylabel("Temps de réponse (s)")
    plt.title("Évolution du Temps de Réponse du Chatbot")
    plt.legend()
    plt.grid(True)
    path = os.path.join(OUTPUT_DIR, "response_time.png")
    plt.savefig(path)
    plt.close()
    return path


def generate_and_display_graphs():
    """Charge les données, génère les graphiques et les affiche dans Streamlit."""
    df = load_data()
    if df is None or df.empty:
        st.warning("❌ Aucune donnée à afficher.")
        return

    # Ajout de l'index pour suivre l'évolution temporelle
    df.reset_index(inplace=True)

    cosine_path = plot_cosine_similarity_evolution(df)
    rouge_path = plot_rouge_scores(df)
    response_time_path = plot_response_time(df)

    st.image(cosine_path, caption="Évolution de la Similarité Cosinus",
             use_container_width=True)
    st.image(rouge_path, caption="Évolution des Scores ROUGE",
             use_container_width=True)
    st.image(response_time_path,
             caption="Évolution du Temps de Réponse", use_container_width=True)
