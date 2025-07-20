import sqlite3
import os
from datetime import datetime
import logging

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'db', 'rag_log.db')

def init_database():    
    try:    
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

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
        logging.error(f"Database error: {str(e)}", exc_info=True)


def log_query(question: str, answer: str):

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
        logging.error(f"Failed to log query: {str(e)}", exc_info=True)

if __name__ == '__main__':    
    print(f"Initializing database at: {DB_PATH}")
    init_database()    
    log_query("This is a test question.", "This is a test answer.")
    print("Test log entry added.")