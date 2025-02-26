import pandas as pd
import psycopg2
from config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD

# Connexion à la base de données PostgreSQL
conn = psycopg2.connect(
    dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT
)
cursor = conn.cursor()

# Charger le CSV correctement en UTF-8
df = pd.read_csv("C:/Users/Aycha/Desktop/dataset_utf8.csv", encoding="utf-8")

# Insérer ligne par ligne
for _, row in df.iterrows():
    try:
        cursor.execute(
            "INSERT INTO ae_qa_table (question, answer, source, focus_area) VALUES (%s, %s, %s, %s)",
            (row["question"], row["answer"], row["source"], row["focus_area"])
        )
    except Exception as e:
        print(f"Erreur d'insertion pour la ligne : {row}")
        print(e)

# Valider et fermer la connexion
conn.commit()
cursor.close()
conn.close()

print("Import terminé avec succès !")
