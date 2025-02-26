""" 
this script tests the connection 
to the PostgreSQL database
"""
import psycopg2
from config import DB_USER, DB_NAME, DB_HOST, DB_PORT, DB_PASSWORD


try:
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    print("Connexion PostgreSQL réussie !")
    conn.close()
except Exception as e:
    print("Erreur de connexion : ", e)
