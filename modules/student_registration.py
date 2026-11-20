
from pathlib import Path
import secrets
import re
import shutil

import cv2
import numpy as np

from modules.database_manager import get_connection
from modules.face_recognition import FaceRecognition
from modules.qr_manager import generate_qr_code


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
STUDENT_PHOTOS_DIR = BASE_DIR / "images" / "student_photos"


def register_student(
    student_id,
    first_name,
    last_name,
    grade,
    major,
    class_name,
    images
):
    """
    Register a real student and save multiple face templates.

    Parameters
    ----------
    images : list
        A list of OpenCV BGR images (NumPy arrays).
        At least 3 images are required.
    """

    # --------------------------------
    # 1. Validate student information
    # --------------------------------

    fields = {
        "student_id": student_id,
        "first_name": first_name,
        "last_name": last_name,
        "grade": grade,
        "major": major,
        "class_name": class_name
    }

    for field_name, value in fields.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} cannot be empty.")

    student_id = student_id.strip()

    # Prevent unsafe folder names
    if not re.fullmatch(r"[A-Za-z0-9_-]+", student_id):
        raise ValueError(
            "Student ID may contain only letters, numbers, "
            "underscores and hyphens."
        )

    if not isinstance(images, (list, tuple)) or len(images) < 3:
        raise ValueError("At least 3 face images are required.")

    # --------------------------------
    # 2. Check duplicate student ID
    # --------------------------------

    connection = get_connection()

    try:
        cursor = connection.cursor()

        cursor.execute(
            "SELECT student_id FROM students WHERE student_id = ?",
            (student_id,)
        )

        if cursor.fetchone():
            raise ValueError("This student ID already exists.")

    finally:
        connection.close()

    # --------------------------------
    # 3. Extract face features
    # --------------------------------

    face_engine = FaceRecognition()

    extracted_features = []

    for index, image in enumerate(images, start=1):

        if not isinstance(image, np.ndarray):
            raise ValueError(
                f"Image {index} is not a valid NumPy array."
            )

        if image.size == 0 or image.ndim != 3:
            raise ValueError(
                f"Image {index} is invalid."
            )

        # YuNet detects the face.
        # SFace extracts its 128-dimensional feature.
        feature = face_engine.extract_feature(image)

        feature = np.asarray(
            feature,
            dtype=np.float32
        ).reshape(-1)

        if feature.size != 128:
            raise ValueError(
                f"Unexpected feature size in image {index}."
            )

        extracted_features.append(feature)

    # --------------------------------
    # 4. Create student photo folder
    # --------------------------------

    student_folder = STUDENT_PHOTOS_DIR / student_id

    if student_folder.exists():
        raise FileExistsError(
            "A photo folder for this student already exists."
        )

    student_folder.mkdir(parents=True, exist_ok=False)

    saved_photo_paths = []

    try:

        # --------------------------------
        # 5. Save images
        # --------------------------------

        for index, image in enumerate(images, start=1):

            photo_path = (
                student_folder / f"photo_{index:02d}.jpg"
            )

            success = cv2.imwrite(
                str(photo_path),
                image
            )

            if not success:
                raise IOError(
                    f"Could not save image {index}."
                )

            saved_photo_paths.append(
                str(photo_path.relative_to(BASE_DIR))
            )

        # --------------------------------
        # 6. Generate secure QR token
        # --------------------------------

        qr_token = secrets.token_urlsafe(24)

        # --------------------------------
        # 7. Save student and face templates
        # --------------------------------

        connection = get_connection()

        try:
            cursor = connection.cursor()

            cursor.execute("""
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
            """, (
                student_id,
                first_name.strip(),
                last_name.strip(),
                grade.strip(),
                major.strip(),
                class_name.strip(),
                qr_token,
                saved_photo_paths[0]
            ))

            for feature, photo_path in zip(
                extracted_features,
                saved_photo_paths
            ):

                cursor.execute("""
                    INSERT INTO face_templates (
                        student_id,
                        feature,
                        photo_path
                    )
                    VALUES (?, ?, ?)
                """, (
                    student_id,
                    feature.tobytes(),
                    photo_path
                ))

            connection.commit()

        except Exception:
            connection.rollback()
            raise

        finally:
            connection.close()

        # --------------------------------
        # 8. Generate QR code
        # --------------------------------

        qr_path = generate_qr_code(student_id)

        return {
            "status": "SUCCESS",
            "student_id": student_id,
            "number_of_photos": len(saved_photo_paths),
            "number_of_templates": len(extracted_features),
            "qr_path": str(qr_path)
        }

    except Exception:

        # Remove saved photos if registration fails.
        # If the database insert was committed, remove
        # the student too (face_templates are cascaded).

        cleanup_connection = get_connection()

        try:
            cleanup_connection.execute(
                "DELETE FROM students WHERE student_id = ?",
                (student_id,)
            )
            cleanup_connection.commit()

        finally:
            cleanup_connection.close()

        if student_folder.exists():
            shutil.rmtree(student_folder)

        raise
