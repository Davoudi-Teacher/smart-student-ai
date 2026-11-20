
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


def get_term_averages(student_id):
    """Calculate the average score for each academic term."""

    connection = get_connection()

    try:
        cursor = connection.cursor()

        cursor.execute("""
            SELECT
                term,
                ROUND(AVG(score), 2) AS average_score
            FROM grades
            WHERE student_id = ?
            GROUP BY term
        """, (student_id,))

        results = [dict(row) for row in cursor.fetchall()]

        term_order = [
            "مستمر اول",
            "پایانی اول",
            "مستمر دوم",
            "پایانی دوم"
        ]

        results.sort(
            key=lambda item: (
                term_order.index(item["term"])
                if item["term"] in term_order
                else len(term_order)
            )
        )

        return results

    finally:
        connection.close()


def analyze_performance(subject_averages, term_averages):
    """Generate academic insights and educational recommendations."""

    strong_subjects = []
    weak_subjects = []
    recommendations = []

    for item in subject_averages:

        subject = item["subject_name"]
        average = item["average_score"]

        if average >= 17:
            strong_subjects.append(subject)

        elif average < 12:
            weak_subjects.append(subject)

    if strong_subjects:
        recommendations.append(
            "عملکرد دانش‌آموز در درس‌های "
            + "، ".join(strong_subjects)
            + " مطلوب است. حفظ این روند توصیه می‌شود."
        )

    if weak_subjects:
        recommendations.append(
            "برای بهبود عملکرد در درس‌های "
            + "، ".join(weak_subjects)
            + " برنامه تمرینی و آموزشی بیشتری در نظر گرفته شود."
        )

    if not subject_averages:
        recommendations.append(
            "برای ارائه تحلیل دقیق‌تر، ابتدا نمرات دانش‌آموز ثبت شوند."
        )

    trend = "اطلاعات کافی برای بررسی روند وجود ندارد."
    trend_change = None

    if len(term_averages) >= 2:

        first_average = term_averages[0]["average_score"]
        last_average = term_averages[-1]["average_score"]

        trend_change = round(last_average - first_average, 2)

        if trend_change >= 0.5:
            trend = "روند پیشرفت تحصیلی"

        elif trend_change <= -0.5:
            trend = "روند افت تحصیلی"

        else:
            trend = "وضعیت تحصیلی تقریباً ثابت"

    if trend == "روند پیشرفت تحصیلی":
        recommendations.append(
            "روند نمرات نشان‌دهنده پیشرفت است. "
            "ادامه روش‌های آموزشی مؤثر توصیه می‌شود."
        )

    elif trend == "روند افت تحصیلی":
        recommendations.append(
            "کاهش میانگین نمرات مشاهده شده است. "
            "بررسی علت افت و گفت‌وگو با دانش‌آموز توصیه می‌شود."
        )

    elif trend == "وضعیت تحصیلی تقریباً ثابت":
        recommendations.append(
            "برای بهبود عملکرد، استفاده از تمرین‌های تکمیلی توصیه می‌شود."
        )

    return {
        "strong_subjects": strong_subjects,
        "weak_subjects": weak_subjects,
        "term_averages": term_averages,
        "trend": trend,
        "trend_change": trend_change,
        "recommendations": recommendations
    }


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
    """Create a complete academic report with performance analysis."""

    student = get_student_profile(student_id)

    if student is None:
        return None

    # دریافت اطلاعات تحصیلی
    grades = get_student_grades(student_id)
    subject_averages = get_subject_averages(student_id)
    attendance = get_attendance_summary(student_id)

    # محاسبه میانگین کل
    if grades:
        overall_average = round(
            sum(item["score"] for item in grades) / len(grades), 2
        )
    else:
        overall_average = 0

    # تعیین وضعیت تحصیلی
    if overall_average >= 17:
        academic_status = "عالی"
    elif overall_average >= 14:
        academic_status = "خوب"
    elif overall_average >= 10:
        academic_status = "متوسط"
    else:
        academic_status = "نیازمند تلاش بیشتر"

    # تعیین وضعیت حضور و غیاب
    if attendance["attendance_percentage"] >= 90:
        attendance_status = "مطلوب"
    elif attendance["attendance_percentage"] >= 75:
        attendance_status = "قابل قبول"
    else:
        attendance_status = "نیازمند پیگیری"

    # تحلیل روند نمرات
    term_averages = get_term_averages(student_id)

    performance_analysis = analyze_performance(
        subject_averages,
        term_averages
    )

    # گزارش نهایی
    return {
        "student": student,
        "grades": grades,
        "subject_averages": subject_averages,
        "attendance": attendance,
        "overall_average": overall_average,
        "academic_status": academic_status,
        "attendance_status": attendance_status,

        # اطلاعات تحلیل هوشمند
        "term_averages": term_averages,
        "performance_analysis": performance_analysis
    }

    