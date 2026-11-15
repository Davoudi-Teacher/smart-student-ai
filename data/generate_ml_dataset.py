
from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_PATH = BASE_DIR / "data" / "ml_training_data.csv"

rng = np.random.default_rng(42)
NUMBER_OF_SAMPLES = 3000

# 1. Generate input features
ability = np.clip(
    rng.normal(14.2, 2.8, NUMBER_OF_SAMPLES), 5, 20
)

study_hours = rng.uniform(1, 8, NUMBER_OF_SAMPLES)

attendance = (
    rng.normal(87, 9, NUMBER_OF_SAMPLES)
    + (ability - 14) * 0.7
)
attendance = np.clip(attendance, 55, 100)

absence_days = np.round((100 - attendance) / 5).astype(int)
late_days = rng.poisson(1.5, NUMBER_OF_SAMPLES)

assignment_rate = (
    65 + ability * 1.5 + study_hours * 1.5
    + rng.normal(0, 8, NUMBER_OF_SAMPLES)
)
assignment_rate = np.clip(assignment_rate, 30, 100)

previous_average = np.clip(
    ability + rng.normal(0, 1.8, NUMBER_OF_SAMPLES),
    0, 20
)

# 2. Calculate a simulated final score

# 2. Calculate a simulated final score

raw_score = (
    0.35 * previous_average
    + 0.20 * (study_hours * 2.5)
    + 0.15 * (attendance / 5)
    + 0.15 * (assignment_rate / 5)
    + 0.15 * (20 - absence_days * 0.5)
    + rng.normal(0, 1.2, NUMBER_OF_SAMPLES)
)

# Adjust score distribution while preserving
# the relationship with input features
final_score = (
    13.5
    + ((raw_score - raw_score.mean()) / raw_score.std()) * 4.0
    + rng.normal(0, 0.5, NUMBER_OF_SAMPLES)
)

final_score = np.clip(final_score, 0, 20)
final_score = np.round(final_score, 2)


# 3. Assign academic labels
def classify_status(score):
    if score >= 17:
        return "عالی"
    elif score >= 14:
        return "خوب"
    elif score >= 10:
        return "متوسط"
    else:
        return "نیازمند تلاش بیشتر"

academic_status = [
    classify_status(score) for score in final_score
]

# 4. Create dataset
df = pd.DataFrame({
    "study_hours": np.round(study_hours, 2),
    "attendance_percentage": np.round(attendance, 2),
    "absence_days": absence_days,
    "late_days": late_days,
    "assignment_rate": np.round(assignment_rate, 2),
    "previous_average": np.round(previous_average, 2),
    "academic_status": academic_status
})

# 5. Save dataset
df.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig"
)

# 6. Display results
print("=" * 45)
print("ML DATASET GENERATED SUCCESSFULLY")
print("=" * 45)

print("Number of samples:", len(df))

print("\nClass Distribution:")
print(df["academic_status"].value_counts())

print("\nClass Percentages:")
print(
    (df["academic_status"].value_counts(normalize=True) * 100)
    .round(2)
)

print("\nSaved to:")
print(OUTPUT_PATH)
