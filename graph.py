"""Module pour générer et afficher des graphiques des métriques du modèle."""
import os
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st




# Chemin du fichier CSV et du dossier de sortie
FILE_PATH = "metrics_log.csv"
OUTPUT_DIR = "graphs"


# Création du dossier de sortie s'il n'existe pas
os.makedirs(OUTPUT_DIR, exist_ok=True)




def load_data() -> pd.DataFrame:
    """
    Charge les données du fichier CSV.


    Returns:
        pd.DataFrame: Le dataframe contenant les métriques du modèle.
    """
    try:
        return pd.read_csv(FILE_PATH)
    except FileNotFoundError:
        st.error("❌ Le fichier de métriques n'a pas été trouvé.")
        return pd.DataFrame()  # Retourne un DataFrame vide pour éviter les erreurs




def plot_cosine_similarity_evolution(df: pd.DataFrame) -> str:
    """
    Génère et sauvegarde l'évolution de la similarité cosinus par requête.


    Args:
        df (pd.DataFrame): Le dataframe contenant les données.


    Returns:
        str: Le chemin de l'image générée.
    """
    plt.figure(figsize=(12, 5))
    plt.plot(df.index, df["cosine_similarity"], marker="o",
             linestyle="-", color="blue", label="Cosine Similarity")
    plt.xlabel("Index de la requête")
    plt.ylabel("Cosine Similarity")
    plt.title("Évolution de la Similarité Cosinus par requête")
    plt.legend()
    plt.grid(True)


    path = os.path.join(OUTPUT_DIR, "cosine_similarity.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path




def plot_response_time(df: pd.DataFrame) -> str:
    """
    Génère et sauvegarde l'évolution du temps de réponse (filtrant > 300s).


    Args:
        df (pd.DataFrame): Le dataframe contenant les données.


    Returns:
        str: Le chemin de l'image générée.
    """
    df_filtered = df[df["response_time"] <= 300]


    plt.figure(figsize=(12, 5))
    plt.plot(df_filtered.index, df_filtered["response_time"], marker="o",
             linestyle="-", color="red", label="Temps de Réponse")
    plt.xlabel("Index de la requête")
    plt.ylabel("Temps d'exécution (s)")
    plt.title("Évolution du Temps d'Exécution (≤ 300s) par requête")
    plt.legend()
    plt.grid(True)


    path = os.path.join(OUTPUT_DIR, "response_time.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path




def plot_rouge_means(df: pd.DataFrame) -> str:
    """
    Génère et sauvegarde l'histogramme des moyennes des scores ROUGE.


    Args:
        df (pd.DataFrame): Le dataframe contenant les données.


    Returns:
        str: Le chemin de l'image générée.
    """
    rouge_means = {
        "ROUGE-1": df["rouge1"].mean(),
        "ROUGE-2": df["rouge2"].mean(),
        "ROUGE-L": df["rougeL"].mean(),
    }


    plt.figure(figsize=(8, 5))
    plt.bar(rouge_means.keys(), rouge_means.values(),
            color=["blue", "red", "green"])
    plt.xlabel("Métrique ROUGE")
    plt.ylabel("Moyenne du score")
    plt.title("Moyenne des Scores ROUGE")
    plt.grid(axis="y", linestyle="--", alpha=0.7)


    # Ajouter les valeurs sur les barres
    for i, (metric, value) in enumerate(rouge_means.items()):
        plt.text(i, value + 0.02, f"{value:.3f}", ha="center",
                 fontsize=12, fontweight="bold")


    path = os.path.join(OUTPUT_DIR, "rouge_means.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path




def generate_and_display_graphs():
    """
    Charge les données, génère les graphiques et les affiche dans Streamlit.
    """
    df = load_data()
    if df.empty:
        st.warning("❌ Aucune donnée à afficher.")
        return


    # Ajout de l'index pour suivre l'évolution temporelle
    df.reset_index(inplace=True)


    cosine_path = plot_cosine_similarity_evolution(df)
    response_time_path = plot_response_time(df)
    rouge_means_path = plot_rouge_means(df)


    st.image(cosine_path, caption="Évolution de la Similarité Cosinus",
             use_container_width=True)
    st.image(response_time_path,
             caption="Évolution du Temps de Réponse (≤ 300s)", use_container_width=True)
    st.image(rouge_means_path, caption="Moyenne des Scores ROUGE",
             use_container_width=True)
    