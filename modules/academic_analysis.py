
from modules.database_manager import get_connection


def get_student_profile(student_id):
    """Retrieve basic student information."""

    connection = get_connection()

    try:
        cursor = connection.cursor()

        cursor.execute("""
            SELECT student_id, first_name, last_name,
                   grade, major, class_name
            FROM students
            WHERE student_id = ?
        """, (student_id,))

        student = cursor.fetchone()

        if student is None:
            return None

        return dict(student)

    finally:
        connection.close()


def get_student_grades(student_id):
    """Retrieve all grades for a student."""

    connection = get_connection()

    try:
        cursor = connection.cursor()

        cursor.execute("""
            SELECT
                subjects.subject_name,
                grades.term,
                grades.assessment,
                grades.score,
                grades.exam_date
            FROM grades
            JOIN subjects
                ON grades.subject_id = subjects.subject_id
            WHERE grades.student_id = ?
            ORDER BY subjects.subject_name, grades.exam_date
        """, (student_id,))

        return [dict(row) for row in cursor.fetchall()]

    finally:
        connection.close()


def get_subject_averages(student_id):
    """Calculate average score for each subject."""

    connection = get_connection()

    try:
        cursor = connection.cursor()

        cursor.execute("""
            SELECT
                subjects.subject_name,
                ROUND(AVG(grades.score), 2) AS average_score
            FROM grades
            JOIN subjects
                ON grades.subject_id = subjects.subject_id
            WHERE grades.student_id = ?
            GROUP BY subjects.subject_id, subjects.subject_name
            ORDER BY subjects.subject_name
        """, (student_id,))

        return [dict(row) for row in cursor.fetchall()]

    finally:
        connection.close()


def get_attendance_summary(student_id):
    """Calculate attendance statistics."""

    connection = get_connection()

    try:
        cursor = connection.cursor()

        cursor.execute("""
            SELECT
                COUNT(*) AS total_days,
                SUM(CASE WHEN status = 'present' THEN 1 ELSE 0 END)
                    AS present_days,
                SUM(CASE WHEN status = 'absent' THEN 1 ELSE 0 END)
                    AS absent_days,
                SUM(CASE WHEN status = 'late' THEN 1 ELSE 0 END)
                    AS late_days
            FROM attendance
            WHERE student_id = ?
        """, (student_id,))

        result = dict(cursor.fetchone())

        total = result["total_days"] or 0
        present = result["present_days"] or 0

        result["attendance_percentage"] = (
            round((present / total) * 100, 2)
            if total > 0 else 0
        )

        return result

    finally:
        connection.close()


def analyze_student(student_id):
    """Create a complete academic report."""

    student = get_student_profile(student_id)

    if student is None:
        return None

    grades = get_student_grades(student_id)
    subject_averages = get_subject_averages(student_id)
    attendance = get_attendance_summary(student_id)

    if grades:
        overall_average = round(
            sum(item["score"] for item in grades) / len(grades), 2
        )
    else:
        overall_average = 0

    if overall_average >= 17:
        academic_status = "عالی"
    elif overall_average >= 14:
        academic_status = "خوب"
    elif overall_average >= 10:
        academic_status = "متوسط"
    else:
        academic_status = "نیازمند تلاش بیشتر"

    if attendance["attendance_percentage"] >= 90:
        attendance_status = "مطلوب"
    elif attendance["attendance_percentage"] >= 75:
        attendance_status = "قابل قبول"
    else:
        attendance_status = "نیازمند پیگیری"

    return {
        "student": student,
        "grades": grades,
        "subject_averages": subject_averages,
        "attendance": attendance,
        "overall_average": overall_average,
        "academic_status": academic_status,
        "attendance_status": attendance_status
    }


if __name__ == "__main__":
    report = analyze_student("10001")

    if report is None:
        print("Student not found.")

    else:
        print("Academic analysis completed.")
        print("Student ID:", report["student"]["student_id"])
        print("Overall Average:", report["overall_average"])
        print("Academic Status:", report["academic_status"])
        print("Attendance:", report["attendance"])
        print("Attendance Status:", report["attendance_status"])

        print("\nSubject Averages:")

        for item in report["subject_averages"]:
            print(
                item["subject_name"],
                ":", item["average_score"]
            )
