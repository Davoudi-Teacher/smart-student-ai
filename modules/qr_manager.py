
from pathlib import Path

import cv2
import qrcode

from modules.database_manager import get_connection


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
QR_DIR = BASE_DIR / "images" / "qr_codes"


def generate_qr_code(student_id):
    """
    Generate a QR Code for one student.
    """

    connection = get_connection()

    try:
        cursor = connection.cursor()

        cursor.execute("""
            SELECT student_id, qr_token
            FROM students
            WHERE student_id = ?
        """, (student_id,))

        student = cursor.fetchone()

        if student is None:
            print("Student not found.")
            return None

        token = student["qr_token"]

    finally:
        connection.close()

    QR_DIR.mkdir(parents=True, exist_ok=True)

    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=10,
        border=4
    )

    qr.add_data(token)
    qr.make(fit=True)

    image = qr.make_image(
        fill_color="black",
        back_color="white"
    )

    output_path = QR_DIR / f"{student_id}.png"

    image.save(output_path)

    print(f"QR Code created for student: {student_id}")

    return output_path


def generate_all_qr_codes():
    """
    Generate QR Codes for all students.
    """

    connection = get_connection()

    try:
        cursor = connection.cursor()

        cursor.execute("""
            SELECT student_id
            FROM students
        """)

        students = cursor.fetchall()

    finally:
        connection.close()

    for student in students:
        generate_qr_code(student["student_id"])

    print(f"\n{len(students)} QR Codes generated successfully.")


def read_qr_code(image):
    """
    Read and decode a QR Code from an image.
    """

    if image is None:
        return None

    detector = cv2.QRCodeDetector()

    decoded_text, points, _ = detector.detectAndDecode(image)

    if decoded_text:
        return decoded_text

    return None


def get_student_by_qr(token):
    """
    Find a student using the decoded QR token.
    """

    if not token:
        return None

    connection = get_connection()

    try:
        cursor = connection.cursor()

        cursor.execute("""
            SELECT
                student_id,
                first_name,
                last_name,
                grade,
                major,
                class_name
            FROM students
            WHERE qr_token = ?
        """, (token,))

        student = cursor.fetchone()

        if student is None:
            return None

        return dict(student)

    finally:
        connection.close()


if __name__ == "__main__":
    generate_all_qr_codes()
