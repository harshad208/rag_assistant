# In app/database_utils.py

import sqlite3
import os
from datetime import datetime

# --- Configuration ---
DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'db', 'rag_log.db')

def init_database():
    """Initializes the SQLite database and creates the log table if it doesn't exist."""
    try:
        # The connect function will create the file if it doesn't exist
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Create table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS query_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL
            )
        ''')
        conn.commit()
        conn.close()
        print("Database initialized successfully.")
    except sqlite3.Error as e:
        print(f"Database error: {e}")


def log_query(question: str, answer: str):
    """Logs a question and its corresponding answer to the database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        cursor.execute('''
            INSERT INTO query_log (timestamp, question, answer)
            VALUES (?, ?, ?)
        ''', (timestamp, question, answer))

        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        print(f"Failed to log query: {e}")

if __name__ == '__main__':
    # This allows you to initialize the database from the command line
    print(f"Initializing database at: {DB_PATH}")
    init_database()
    # Example log entry
    log_query("This is a test question.", "This is a test answer.")
    print("Test log entry added.")