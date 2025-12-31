import sqlite3
from config.settings import DB_PATH

def get_connection():
    return sqlite3.connect(DB_PATH)

def init_db():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS cluster_capacity (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL,
        cluster_name TEXT NOT NULL,
        cpu_used_ghz REAL,
        cpu_total_ghz REAL,
        mem_used_gb REAL,
        mem_total_gb REAL,
        storage_used_tb REAL,
        storage_total_tb REAL
    )
    """)

    cur.execute("""
    CREATE VIEW IF NOT EXISTS cluster_capacity_units AS
    SELECT
        ts,
        cluster_name,
        cpu_used_ghz AS cpu_used_cores,
        cpu_total_ghz AS cpu_total_cores,
        mem_used_gb AS mem_used_gib,
        mem_total_gb AS mem_total_gib,
        storage_used_tb AS storage_used_tib,
        storage_total_tb AS storage_total_tib
    FROM cluster_capacity
    """)

    conn.commit()
    conn.close()
