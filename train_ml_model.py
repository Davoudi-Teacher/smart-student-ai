
from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score
)

import matplotlib.pyplot as plt


# -----------------------------------------
# 1. Define project paths
# -----------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "ml_training_data.csv"

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "student_status_model.joblib"


# -----------------------------------------
# 2. Load dataset
# -----------------------------------------

df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")

print("=" * 55)
print("STUDENT ACADEMIC STATUS - RANDOM FOREST")
print("=" * 55)

print("\nDataset shape:", df.shape)


# -----------------------------------------
# 3. Separate features and target
# -----------------------------------------

X = df.drop(columns=["academic_status"])
y = df["academic_status"]

print("\nInput features:")
print(X.columns.tolist())


# -----------------------------------------
# 4. Split training and testing data
# -----------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# -----------------------------------------
# 5. Create Random Forest model
# -----------------------------------------

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)


# -----------------------------------------
# 6. Train the model
# -----------------------------------------

print("\nTraining model...")

model.fit(X_train, y_train)

print("Model training completed!")


# -----------------------------------------
# 7. Make predictions
# -----------------------------------------

y_pred = model.predict(X_test)


# -----------------------------------------
# 8. Evaluate model
# -----------------------------------------

accuracy = accuracy_score(y_test, y_pred)

macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro"
)

weighted_f1 = f1_score(
    y_test,
    y_pred,
    average="weighted"
)

print("\n" + "=" * 55)
print("MODEL EVALUATION")
print("=" * 55)

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Macro F1 Score: {macro_f1:.4f}")
print(f"Weighted F1 Score: {weighted_f1:.4f}")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# -----------------------------------------
# 9. Confusion Matrix
# -----------------------------------------

labels = model.classes_

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

print("\nConfusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels
)

fig, ax = plt.subplots(figsize=(9, 7))

disp.plot(
    cmap="Blues",
    values_format="d",
    xticks_rotation=25,
    ax=ax
)

plt.title("Random Forest - Confusion Matrix")
plt.tight_layout()

plt.savefig(
    MODEL_DIR / "confusion_matrix.png",
    dpi=200
)

plt.show()


# -----------------------------------------
# 10. Feature Importance
# -----------------------------------------

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(importance.to_string(index=False))

plt.figure(figsize=(9, 5))

plt.barh(
    importance["Feature"],
    importance["Importance"]
)

plt.xlabel("Importance")
plt.title("Feature Importance - Random Forest")
plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(
    MODEL_DIR / "feature_importance.png",
    dpi=200
)

plt.show()


# -----------------------------------------
# 11. Save trained model
# -----------------------------------------


# -----------------------------------------
# 11. Save trained model
# -----------------------------------------

import joblib

MODEL_DIR.mkdir(parents=True, exist_ok=True)

joblib.dump(model, MODEL_PATH)

print("\n" + "=" * 55)
print("MODEL SAVED SUCCESSFULLY")
print("=" * 55)

print("Model path:")
print(MODEL_PATH)

print("File exists:", MODEL_PATH.exists())

# -----------------------------------------
# 12. Save feature importance
# -----------------------------------------

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(importance.to_string(index=False))

plt.figure(figsize=(9, 5))

plt.barh(
    importance["Feature"],
    importance["Importance"]
)

plt.xlabel("Importance")
plt.title("Feature Importance - Random Forest")
plt.gca().invert_yaxis()

plt.tight_layout()

FEATURE_PATH = MODEL_DIR / "feature_importance.png"

plt.savefig(FEATURE_PATH, dpi=200)
plt.close()

print("\nFeature importance chart saved:")
print(FEATURE_PATH)

print("\nTraining process completed successfully!")

