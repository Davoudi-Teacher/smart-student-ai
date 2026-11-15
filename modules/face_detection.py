
import cv2
import numpy as np


# Load OpenCV Haar Cascade face detector
CASCADE_PATH = (
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)


def check_image_quality(image):
    """
    Check basic image quality:
    brightness, sharpness and dimensions.
    """

    if image is None or image.size == 0:
        return {
            "valid": False,
            "message": "تصویر معتبر نیست."
        }

    height, width = image.shape[:2]

    if width < 100 or height < 100:
        return {
            "valid": False,
            "message": "ابعاد تصویر بسیار کوچک است."
        }

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    brightness = float(np.mean(gray))

    sharpness = float(
        cv2.Laplacian(gray, cv2.CV_64F).var()
    )

    quality_messages = []

    if brightness < 40:
        quality_messages.append("تصویر تاریک است.")

    elif brightness > 220:
        quality_messages.append("تصویر بیش از حد روشن است.")

    if sharpness < 50:
        quality_messages.append("تصویر احتمالاً تار است.")

    return {
        "valid": True,
        "width": width,
        "height": height,
        "brightness": round(brightness, 2),
        "sharpness": round(sharpness, 2),
        "quality_messages": quality_messages
    }


def detect_faces(image):
    """
    Detect faces and draw bounding boxes.

    Input image must be in BGR format.
    This function detects faces, not identities.
    """

    if image is None:
        raise ValueError("تصویر ورودی معتبر نیست.")

    if face_cascade.empty():
        raise RuntimeError(
            "OpenCV face detection model could not be loaded."
        )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40)
    )

    result_image = image.copy()

    for (x, y, width, height) in faces:

        cv2.rectangle(
            result_image,
            (x, y),
            (x + width, y + height),
            (0, 255, 0),
            3
        )

        cv2.putText(
            result_image,
            "Face detected",
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    return {
        "face_count": len(faces),
        "faces": faces,
        "image": result_image
    }
