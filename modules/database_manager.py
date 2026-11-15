
import sqlite3
from pathlib import Path


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "database" / "students.db"


def get_connection():
    """
    Create and return a connection to the SQLite database.
    """

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row

    # Enable foreign key constraints
    connection.execute("PRAGMA foreign_keys = ON")

    return connection


def create_tables():
    """
    Create all required database tables.
    """

    connection = get_connection()
    cursor = connection.cursor()

    # Students table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            student_id TEXT PRIMARY KEY,
            first_name TEXT NOT NULL,
            last_name TEXT NOT NULL,
            grade TEXT NOT NULL,
            major TEXT NOT NULL,
            class_name TEXT NOT NULL,
            qr_token TEXT UNIQUE NOT NULL,
            photo_path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Subjects table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS subjects (
            subject_id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject_name TEXT UNIQUE NOT NULL
        )
    """)

    # Grades table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS grades (
            grade_id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            subject_id INTEGER NOT NULL,
            term TEXT NOT NULL,
            assessment TEXT NOT NULL,
            score REAL NOT NULL CHECK(score >= 0 AND score <= 20),
            exam_date TEXT,

            FOREIGN KEY (student_id)
                REFERENCES students(student_id),

            FOREIGN KEY (subject_id)
                REFERENCES subjects(subject_id)
        )
    """)

    # Attendance table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            attendance_id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            attendance_date TEXT NOT NULL,
            status TEXT NOT NULL
                CHECK(status IN ('present', 'absent', 'late')),

            FOREIGN KEY (student_id)
                REFERENCES students(student_id),

            UNIQUE(student_id, attendance_date)
        )
    """)

    connection.commit()
    connection.close()

    print("Database tables created successfully!")


def get_database_info():
    """
    Display the number of records in each table.
    """

    connection = get_connection()
    cursor = connection.cursor()

    tables = [
        "students",
        "subjects",
        "grades",
        "attendance"
    ]

    print("\nDatabase Information")
    print("-" * 35)

    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]

        print(f"{table}: {count} records")

    print("-" * 35)

    connection.close()


if __name__ == "__main__":

    create_tables()
    get_database_info()
