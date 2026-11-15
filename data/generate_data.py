
import sys
import random
from pathlib import Path
from datetime import date, timedelta

import pandas as pd

# Add project root to Python path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from modules.database_manager import create_tables, get_connection


# Reproducible random data
random.seed(42)

NUMBER_OF_STUDENTS = 30

SUBJECTS = [
    "ریاضی",
    "برنامه‌نویسی",
    "هوش مصنوعی",
    "پایگاه داده",
    "شبکه‌های کامپیوتری"
]

ASSESSMENTS = [
    ("نوبت اول", "آزمون اول", "2026-10-03"),
    ("نوبت اول", "آزمون دوم", "2026-10-10"),
    ("نوبت اول", "آزمون سوم", "2026-10-17")
]


def clear_demo_data(connection):
    """Delete previously generated demo records."""

    cursor = connection.cursor()

    cursor.execute("DELETE FROM attendance")
    cursor.execute("DELETE FROM grades")
    cursor.execute("DELETE FROM students")
    cursor.execute("DELETE FROM subjects")

    # Reset automatic IDs
    cursor.execute("DELETE FROM sqlite_sequence")

    print("Previous demo data cleared.")


def generate_students(connection):
    """Insert synthetic student records."""

    cursor = connection.cursor()

    students = []

    for i in range(1, NUMBER_OF_STUDENTS + 1):

        student_id = f"100{i:02d}"

        first_name = "دانش‌آموز"
        last_name = f"آزمایشی {i:02d}"

        students.append((
            student_id,
            first_name,
            last_name,
            "دهم",
            "شبکه و نرم‌افزار رایانه",
            "۱۰ کامپیوتر - الف",
            f"QR-{student_id}",
            None
        ))

    cursor.executemany("""
        INSERT INTO students (
            student_id,
            first_name,
            last_name,
            grade,
            major,
            class_name,
            qr_token,
            photo_path
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, students)

    print(f"{len(students)} synthetic students created.")


def generate_subjects(connection):
    """Insert school subjects."""

    cursor = connection.cursor()

    for subject in SUBJECTS:
        cursor.execute("""
            INSERT INTO subjects (subject_name)
            VALUES (?)
        """, (subject,))

    print(f"{len(SUBJECTS)} subjects created.")


def generate_grades(connection):
    """Generate three assessment scores for each student and subject."""

    cursor = connection.cursor()

    cursor.execute("SELECT student_id FROM students")
    students = cursor.fetchall()

    cursor.execute("SELECT subject_id, subject_name FROM subjects")
    subjects = cursor.fetchall()

    grade_records = []

    for student in students:

        student_id = student["student_id"]

        student_level = random.uniform(11, 19)

        for subject in subjects:

            subject_adjustment = random.uniform(-2, 2)

            for index, (term, assessment, exam_date) in enumerate(ASSESSMENTS):

                progress = index * random.uniform(0.1, 0.8)

                score = (
                    student_level
                    + subject_adjustment
                    + progress
                    + random.uniform(-2, 2)
                )

                score = round(max(0, min(20, score)), 2)

                grade_records.append((
                    student_id,
                    subject["subject_id"],
                    term,
                    assessment,
                    score,
                    exam_date
                ))

    cursor.executemany("""
        INSERT INTO grades (
            student_id,
            subject_id,
            term,
            assessment,
            score,
            exam_date
        )
        VALUES (?, ?, ?, ?, ?, ?)
    """, grade_records)

    print(f"{len(grade_records)} synthetic grade records created.")


def generate_attendance(connection):
    """Generate 20 days of attendance for every student."""

    cursor = connection.cursor()

    cursor.execute("SELECT student_id FROM students")
    students = cursor.fetchall()

    attendance_records = []

    start_date = date(2026, 10, 1)

    for student in students:

        for day_number in range(20):

            current_date = start_date + timedelta(days=day_number)

            chance = random.random()

            if chance < 0.85:
                status = "present"
            elif chance < 0.95:
                status = "absent"
            else:
                status = "late"

            attendance_records.append((
                student["student_id"],
                current_date.isoformat(),
                status
            ))

    cursor.executemany("""
        INSERT INTO attendance (
            student_id,
            attendance_date,
            status
        )
        VALUES (?, ?, ?)
    """, attendance_records)

    print(
        f"{len(attendance_records)} "
        "synthetic attendance records created."
    )


def export_csv(connection):
    """Export synthetic data to CSV files."""

    students_df = pd.read_sql_query("""
        SELECT
            student_id,
            first_name,
            last_name,
            grade,
            major,
            class_name
        FROM students
    """, connection)

    grades_df = pd.read_sql_query("""
        SELECT
            students.student_id,
            students.first_name,
            students.last_name,
            subjects.subject_name,
            grades.term,
            grades.assessment,
            grades.score,
            grades.exam_date
        FROM grades
        JOIN students
            ON grades.student_id = students.student_id
        JOIN subjects
            ON grades.subject_id = subjects.subject_id
    """, connection)

    students_df.to_csv(
        BASE_DIR / "data" / "sample_students.csv",
        index=False,
        encoding="utf-8-sig"
    )

    grades_df.to_csv(
        BASE_DIR / "data" / "grades.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("CSV files exported successfully.")


def main():

    connection = None

    try:

        create_tables()

        connection = get_connection()

        clear_demo_data(connection)

        generate_students(connection)
        generate_subjects(connection)
        generate_grades(connection)
        generate_attendance(connection)

        connection.commit()

        export_csv(connection)

        print("\nData generation completed!")

    except Exception as error:

        if connection is not None:
            connection.rollback()

        print(f"\nError: {error}")
        raise

    finally:

        if connection is not None:
            connection.close()


if __name__ == "__main__":
    main()
