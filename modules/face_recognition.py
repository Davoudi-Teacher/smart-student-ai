
from pathlib import Path

import cv2
import numpy as np


# مسیر اصلی پروژه
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

YUNET_MODEL = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_MODEL = MODELS_DIR / "face_recognition_sface_2021dec.onnx"


class FaceRecognition:

    def __init__(self, threshold=0.363):

        if not YUNET_MODEL.exists():
            raise FileNotFoundError(f"YuNet model not found: {YUNET_MODEL}")

        if not SFACE_MODEL.exists():
            raise FileNotFoundError(f"SFace model not found: {SFACE_MODEL}")

        self.threshold = threshold

        self.detector = cv2.FaceDetectorYN_create(
            str(YUNET_MODEL),
            "",
            (320, 320)
        )

        self.recognizer = cv2.FaceRecognizerSF_create(
            str(SFACE_MODEL),
            ""
        )

    def extract_feature(self, image):

        if image is None or image.size == 0:
            raise ValueError("Invalid image")

        height, width = image.shape[:2]

        self.detector.setInputSize((width, height))

        _, faces = self.detector.detect(image)

        if faces is None or len(faces) == 0:
            raise ValueError("No face detected")

        if len(faces) > 1:
            raise ValueError("Multiple faces detected")

        aligned_face = self.recognizer.alignCrop(
            image,
            faces[0]
        )

        feature = self.recognizer.feature(aligned_face)

        return feature

    def compare_features(self, feature1, feature2):

        score = self.recognizer.match(
            feature1,
            feature2,
            cv2.FaceRecognizerSF_FR_COSINE
        )

        return float(score)

    def is_same_person(self, feature1, feature2):

        score = self.compare_features(feature1, feature2)

        return score >= self.threshold, score

    
    
    def identify(
        self,
        query_feature,
        student_features,
        score_threshold=0.70,
        margin_threshold=0.02
    ):

        results = []

        for student_id, features in student_features.items():

            if not features:
                continue

            scores = [
                self.compare_features(query_feature, feature)
                for feature in features
            ]

            best_score = max(scores)

            results.append({
                "student_id": student_id,
                "score": best_score
            })

        results.sort(
            key=lambda item: item["score"],
            reverse=True
        )

        if not results:
            return {
                "status": "UNKNOWN",
                "student_id": None,
                "score": 0.0,
                "margin": 0.0
            }

        best = results[0]
        second = results[1] if len(results) > 1 else None

        margin = (
            best["score"] - second["score"]
            if second is not None else 1.0
        )

        if best["score"] < score_threshold:
            status = "UNKNOWN"
            student_id = None

        elif margin < margin_threshold:
            status = "MANUAL_REVIEW"
            student_id = None

        else:
            status = "IDENTIFIED"
            student_id = best["student_id"]

        return {
            "status": status,
            "student_id": student_id,
            "score": best["score"],
            "margin": margin
        }


