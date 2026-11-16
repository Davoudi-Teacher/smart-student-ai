import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import cv2
import numpy as np
import joblib

from PIL import Image

from modules.academic_analysis import analyze_student
from modules.qr_manager import read_qr_code, get_student_by_qr
from modules.face_detection import detect_faces
from modules.database_manager import get_connection, create_tables

from data.generate_data import (
    generate_students,
    generate_subjects,
    generate_grades,
    generate_attendance
)

# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(
    page_title="سامانه هوشمند دانش‌آموزی",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = __import__("pathlib").Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "student_status_model.joblib"
def initialize_database():
    create_tables()

    connection = get_connection()

    try:
        cursor = connection.cursor()

        cursor.execute("SELECT COUNT(*) FROM students")
        students_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM subjects")
        subjects_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM grades")
        grades_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM attendance")
        attendance_count = cursor.fetchone()[0]

        counts = [
            students_count,
            subjects_count,
            grades_count,
            attendance_count
        ]

        # اگر همه جدول‌ها خالی باشند، داده آزمایشی تولید می‌شود
        if all(count == 0 for count in counts):

            generate_students(connection)
            generate_subjects(connection)
            generate_grades(connection)
            generate_attendance(connection)

            connection.commit()

        # اگر بخشی از اطلاعات وجود داشته باشد، از تغییر آن جلوگیری می‌شود
        elif not all(count > 0 for count in counts):

            st.error(
                "پایگاه داده ناقص است. برای جلوگیری از حذف اطلاعات، "
                "تولید خودکار داده انجام نشد."
            )
            st.stop()

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


initialize_database()

STATUS_COLORS = {
    "عالی": "#10B981",
    "خوب": "#3B82F6",
    "متوسط": "#F59E0B",
    "نیازمند تلاش بیشتر": "#F43F5E"
}


# ==========================================================
# LOAD MODEL
# ==========================================================
@st.cache_resource
def load_ml_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


# ==========================================================
# CUSTOM UI / COLORFUL 3D THEME
# ==========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@300;400;500;600;700;800;900&display=swap');

:root {
    --ink: #172554;
    --muted: #64748b;
    --purple: #7657F6;
    --blue: #2585F5;
    --cyan: #18B6C9;
    --pink: #F05A9D;
    --surface: rgba(255,255,255,.94);
    --line: #DCE5F5;
}

html, body, [class*="css"], [data-testid="stApp"] {
    font-family: 'Vazirmatn', Tahoma, sans-serif !important;
}
.stApp {
    direction: rtl;
    text-align: right;
    background:
      radial-gradient(circle at 8% 8%, rgba(118,87,246,.13), transparent 25%),
      radial-gradient(circle at 92% 18%, rgba(24,182,201,.13), transparent 24%),
      linear-gradient(135deg, #F4F7FF 0%, #F8FAFF 48%, #F2F7FF 100%);
    color: var(--ink);
}
.block-container {
    max-width: 1500px;
    padding-top: 1.1rem;
    padding-bottom: 2.5rem;
}
h1, h2, h3, h4, p, label, span, div {
    font-family: 'Vazirmatn', Tahoma, sans-serif;
}
h1, h2, h3, h4, p, label { direction: rtl; text-align: right; }
h1 { color: #172554; font-weight: 900 !important; }
h2, h3 { color: #243B70; font-weight: 800 !important; }
[data-testid="stSidebar"] {
    background: linear-gradient(165deg, #172554 0%, #30266F 52%, #174B76 100%);
    border-left: 1px solid rgba(255,255,255,.15);
}
[data-testid="stSidebar"] * { color: #F8FAFF !important; }
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    background: rgba(255,255,255,.08);
    border: 1px solid rgba(255,255,255,.10);
    border-radius: 14px;
    padding: 10px 12px;
    margin: 5px 0;
    transition: all .2s ease;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: rgba(255,255,255,.18);
    transform: translateX(-3px);
}
[data-testid="stMetric"] {
    background: linear-gradient(145deg, rgba(255,255,255,.98), rgba(239,245,255,.94));
    border: 1px solid rgba(133,157,220,.25);
    border-radius: 20px;
    padding: 17px 18px;
    box-shadow: 0 9px 0 rgba(45,68,130,.06), 0 15px 30px rgba(44,62,120,.09);
    transition: transform .2s ease, box-shadow .2s ease;
    direction: rtl;
    text-align: right;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 0 rgba(45,68,130,.05), 0 20px 35px rgba(44,62,120,.14);
}
[data-testid="stMetricLabel"] { color: #64748B !important; font-weight: 700; }
[data-testid="stMetricValue"] { color: #172554 !important; font-weight: 900; }
.stButton > button, .stDownloadButton > button {
    border: 0;
    border-radius: 13px;
    color: white;
    font-weight: 800;
    background: linear-gradient(110deg, #7657F6, #2585F5 58%, #18B6C9);
    box-shadow: 0 6px 0 rgba(40,67,145,.15), 0 12px 22px rgba(80,91,190,.22);
    transition: all .18s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    color: white;
    transform: translateY(-3px);
    box-shadow: 0 9px 0 rgba(40,67,145,.12), 0 16px 28px rgba(80,91,190,.25);
}
div[data-testid="stForm"] {
    background: rgba(255,255,255,.78);
    border: 1px solid rgba(142,160,215,.25);
    border-radius: 20px;
    padding: 18px;
    box-shadow: 0 12px 28px rgba(44,62,120,.08);
}
.stTabs [data-baseweb="tab-list"] {
    direction: rtl;
    gap: 8px;
    background: rgba(226,234,255,.65);
    border-radius: 16px;
    padding: 7px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 12px;
    font-weight: 700;
}
.stTabs [aria-selected="true"] {
    background: white !important;
    box-shadow: 0 4px 12px rgba(45,68,130,.12);
}
div[data-testid="stDataFrame"] {
    direction: rtl;
    border-radius: 15px;
    overflow: hidden;
    border: 1px solid #DCE5F5;
    background: #FFFFFF !important;
    box-shadow: 0 7px 20px rgba(38,58,110,.07);
}
div[data-testid="stDataFrame"] [data-testid="glideDataEditor"],
div[data-testid="stDataFrame"] canvas {
    background: #FFFFFF !important;
}
[data-testid="stTable"] {
    direction: rtl;
    background: #FFFFFF !important;
    border: 1px solid #DCE5F5;
    border-radius: 14px;
    overflow: hidden;
}
[data-testid="stTable"] table {
    background: #FFFFFF !important;
    color: #24345B !important;
}
[data-testid="stTable"] th {
    background: #EEF3FF !important;
    color: #263B70 !important;
    font-weight: 800 !important;
}
[data-testid="stTable"] td {
    background: #FFFFFF !important;
    color: #334155 !important;
}
div[data-testid="stFileUploader"] {
    background: rgba(255,255,255,.7);
    border: 1px dashed #9AADE0;
    border-radius: 16px;
    padding: 10px;
}
hr { border-color: rgba(120,140,190,.20); }
.hero {
    position: relative;
    overflow: hidden;
    padding: 21px 26px;
    border-radius: 25px;
    color: white;
    background: linear-gradient(115deg, #29245F 0%, #5B45C9 45%, #168DAE 100%);
    box-shadow: 0 10px 0 rgba(37,53,119,.10), 0 22px 42px rgba(57,73,150,.22);
    margin: 4px 0 22px 0;
}
.hero:after {
    content: "";
    position: absolute;
    width: 230px; height: 230px;
    border-radius: 50%;
    left: -55px; top: -105px;
    background: rgba(255,255,255,.10);
    box-shadow: 0 0 0 28px rgba(255,255,255,.05), 0 0 0 58px rgba(255,255,255,.035);
}
.hero h1, .hero p { color: white !important; margin: 0; position: relative; z-index: 1; }
.hero h1 { font-size: clamp(1.45rem, 2.5vw, 2.15rem); line-height: 1.55; }
.hero p { opacity: .94; margin-top: 6px; font-size: .98rem; }
.section-chip {
    display: inline-block;
    color: #5142A7;
    background: #ECEAFF;
    border: 1px solid #DCD7FF;
    border-radius: 999px;
    padding: 5px 12px;
    font-size:  .83rem;
    font-weight: 800;
    margin-bottom: 8px;
}
.glass-card {
    background: rgba(255,255,255,.94);
    border: 1px solid rgba(148,163,220,.25);
    border-radius: 20px;
    padding: 18px;
    box-shadow: 0 8px 0 rgba(45,68,130,.045), 0 16px 30px rgba(44,62,120,.075);
}
div[data-testid="stAlert"] { border-radius: 15px; }
footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
[data-testid="stToolbar"] { background: transparent; }
[data-testid="stHeader"] { background: rgba(248,250,255,.78); }
[data-testid="stDecoration"] { display: none; }
.stCaption, [data-testid="stCaptionContainer"] { color: #64748B !important; }
[data-testid="stAlert"] p { color: inherit; }

/* ---------- High-contrast widget and text fixes ---------- */
.stApp,
[data-testid="stAppViewContainer"] {
    color: #24345B !important;
}

/* Ordinary explanatory text and widget labels */
[data-testid="stMarkdownContainer"] p,
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p,
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] p,
.stMarkdown p,
.stCaption,
label,
small {
    color: #475569 !important;
}

/* Keep text inside gradient hero banners white */
.hero,
.hero h1,
.hero p {
    color: #FFFFFF !important;
}

/* Text inputs, number inputs and select controls */
.stTextInput input,
.stNumberInput input,
.stDateInput input,
.stTimeInput input,
[data-baseweb="input"] input,
[data-baseweb="select"] > div,
[data-baseweb="select"] input,
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input {
    background: #FFFFFF !important;
    color: #1E293B !important;
    -webkit-text-fill-color: #1E293B !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 10px !important;
    opacity: 1 !important;
}
.stTextInput input::placeholder,
[data-baseweb="input"] input::placeholder {
    color: #94A3B8 !important;
    -webkit-text-fill-color: #94A3B8 !important;
    opacity: 1 !important;
}

/* Dropdown menus and options */
[data-baseweb="popover"],
[data-baseweb="menu"],
[role="listbox"],
[role="option"] {
    color: #1E293B !important;
}
[role="option"] {
    background: #FFFFFF !important;
}
[role="option"]:hover {
    background: #EEF2FF !important;
}

/* All action buttons, including form submit buttons */
.stButton > button,
.stDownloadButton > button,
.stFormSubmitButton > button,
[data-testid="stFormSubmitButton"] button,
button[kind="primary"],
button[kind="secondary"] {
    background: linear-gradient(110deg, #6547E8 0%, #287DEB 58%, #079EAF 100%) !important;
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    border: 0 !important;
    font-weight: 800 !important;
    text-shadow: 0 1px 1px rgba(0,0,0,.12) !important;
}
.stButton > button *,
.stDownloadButton > button *,
.stFormSubmitButton > button *,
[data-testid="stFormSubmitButton"] button * {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}
.stButton > button:hover,
.stDownloadButton > button:hover,
.stFormSubmitButton > button:hover,
[data-testid="stFormSubmitButton"] button:hover {
    filter: brightness(1.06);
}

/* File uploader and camera controls */
[data-testid="stFileUploader"],
[data-testid="stCameraInput"] {
    color: #334155 !important;
}
[data-testid="stFileUploader"] section,
[data-testid="stFileUploader"] button {
    color: #334155 !important;
}
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] span {
    color: #475569 !important;
}

/* Alerts: readable text on their own light backgrounds */
[data-testid="stAlert"] p,
[data-testid="stAlert"] span {
    color: #334155 !important;
}

/* Metric labels and values */
[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] * {
    color: #52627D !important;
}
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] * {
    color: #172554 !important;
}

/* Table container should stay light */
[data-testid="stDataFrame"],
[data-testid="stTable"] {
    background: #FFFFFF !important;
    color: #24345B !important;
    border-color: #DCE5F5 !important;
}

/* Active/inactive tabs */
.stTabs [data-baseweb="tab"] {
    color: #475569 !important;
}
.stTabs [aria-selected="true"] {
    color: #4338CA !important;
}


/* ==========================================================
   FINAL CONTRAST TUNING
   Preserve the original dark sidebar palette; improve only
   the readability of the main content and form controls.
   ========================================================== */

/* Main page: dark readable text on a soft, light background */
section.main,
section.main [data-testid="stMarkdownContainer"],
section.main [data-testid="stMarkdownContainer"] p,
section.main [data-testid="stMarkdownContainer"] li,
section.main [data-testid="stCaptionContainer"],
section.main [data-testid="stCaptionContainer"] p,
section.main [data-testid="stWidgetLabel"],
section.main [data-testid="stWidgetLabel"] p,
section.main label,
section.main small {
    color: #263653 !important;
}

/* Headings stand out clearly without changing the hero banner */
section.main h1,
section.main h2,
section.main h3,
section.main h4 {
    color: #172554 !important;
    font-weight: 800 !important;
}

/* Keep the existing colorful gradient hero and its white text */
section.main .hero,
section.main .hero h1,
section.main .hero p {
    color: #FFFFFF !important;
}

/* Section chips: stronger purple text on pale lavender */
section.main .section-chip {
    color: #43358F !important;
    background: #E9E7FF !important;
    border-color: #CFC9FF !important;
}

/* Glass cards and forms */
section.main .glass-card,
section.main div[data-testid="stForm"] {
    background: #FFFFFF !important;
    color: #263653 !important;
    border: 1px solid #D7E0F0 !important;
}

/* Metric labels and values */
section.main [data-testid="stMetricLabel"],
section.main [data-testid="stMetricLabel"] * {
    color: #52627D !important;
}
section.main [data-testid="stMetricValue"],
section.main [data-testid="stMetricValue"] * {
    color: #172554 !important;
}

/* Inputs: white surface, dark entered text, visible border */
section.main input,
section.main textarea,
section.main [data-baseweb="input"] input,
section.main [data-baseweb="textarea"] textarea,
section.main [data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    color: #1E293B !important;
    -webkit-text-fill-color: #1E293B !important;
    border-color: #AEBED8 !important;
    opacity: 1 !important;
}
section.main input::placeholder,
section.main textarea::placeholder {
    color: #7888A3 !important;
    -webkit-text-fill-color: #7888A3 !important;
    opacity: 1 !important;
}

/* Buttons: keep the same vivid blue-purple gradient with white text */
section.main .stButton > button,
section.main .stDownloadButton > button,
section.main [data-testid="stFormSubmitButton"] button,
section.main .stFormSubmitButton > button {
    background: linear-gradient(110deg, #6547E8 0%, #287DEB 58%, #079EAF 100%) !important;
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    border: 0 !important;
    font-weight: 800 !important;
    box-shadow: 0 5px 0 rgba(40,67,145,.15), 0 10px 20px rgba(80,91,190,.18) !important;
}
section.main .stButton > button *,
section.main .stDownloadButton > button *,
section.main [data-testid="stFormSubmitButton"] button * {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}

/* File uploader, camera and dropdown menus */
section.main [data-testid="stFileUploader"],
section.main [data-testid="stCameraInput"],
section.main [data-baseweb="popover"],
section.main [role="listbox"],
section.main [role="option"] {
    color: #263653 !important;
}
section.main [role="option"] {
    background: #FFFFFF !important;
}
section.main [role="option"]:hover {
    background: #EEF2FF !important;
}

/* Alerts and table text */
section.main [data-testid="stAlert"] p,
section.main [data-testid="stAlert"] span {
    color: #263653 !important;
}
section.main [data-testid="stTable"],
section.main [data-testid="stTable"] table,
section.main [data-testid="stTable"] td {
    background: #FFFFFF !important;
    color: #263653 !important;
}
section.main [data-testid="stTable"] th {
    background: #EAF0FF !important;
    color: #243B70 !important;
}

/* IMPORTANT: preserve the original sidebar's deep blue/purple look
   and its bright white navigation text and labels. */
[data-testid="stSidebar"],
[data-testid="stSidebar"] * {
    color: #F8FAFF !important;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] *,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] button,
[data-testid="stSidebar"] button * {
    color: #F8FAFF !important;
    -webkit-text-fill-color: #F8FAFF !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label {
    color: #FFFFFF !important;
    background: rgba(255,255,255,.09) !important;
    border: 1px solid rgba(255,255,255,.16) !important;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: rgba(255,255,255,.18) !important;
}


/* ==========================================================
   TARGETED FIX FOR THE LOW-CONTRAST BOXES MARKED BY THE USER
   ========================================================== */

/* Success / warning summary cards: pale backgrounds + dark text */
section.main .status-success-card,
section.main .status-success-card * {
    background: #E2F5E9 !important;
    color: #14532D !important;
    -webkit-text-fill-color: #14532D !important;
}
section.main .status-warning-card,
section.main .status-warning-card * {
    background: #FFF2D6 !important;
    color: #78350F !important;
    -webkit-text-fill-color: #78350F !important;
}
section.main .status-success-card,
section.main .status-warning-card {
    display: block;
    width: 100%;
    box-sizing: border-box;
    padding: 16px 20px;
    margin: 10px 0;
    border-radius: 14px;
    border: 1px solid transparent;
    font-weight: 800;
    line-height: 1.9;
    text-align: right;
    box-shadow: 0 5px 0 rgba(35,55,100,.07), 0 10px 22px rgba(35,55,100,.07);
}
section.main .status-success-card {
    border-color: #A7DDB8 !important;
}
section.main .status-warning-card {
    border-color: #F2D18A !important;
}

/* The small footer line in the dark sidebar must be bright white */
[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] *,
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] .stCaption * {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    opacity: 1 !important;
    font-weight: 700 !important;
}

/* Keep sidebar's educational notice legible on its dark card */
[data-testid="stSidebar"] [data-testid="stAlert"],
[data-testid="stSidebar"] [data-testid="stAlert"] * {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}
[data-testid="stSidebar"] [data-testid="stAlert"] {
    background: #202D49 !important;
    border: 1px solid rgba(255,255,255,.14) !important;
}

</style>
""", unsafe_allow_html=True)


# ==========================================================
# REUSABLE UI HELPERS
# ==========================================================


def styled_table(df):
    """Apply a high-contrast light style to Streamlit data tables."""
    return (
        df.style
        .set_properties(**{
            "background-color": "#FFFFFF",
            "color": "#24345B",
            "border-color": "#DCE5F5",
            "text-align": "right",
        })
        .set_table_styles([
            {
                "selector": "th",
                "props": [
                    ("background-color", "#EAF0FF"),
                    ("color", "#243B70"),
                    ("font-weight", "800"),
                    ("text-align", "right"),
                    ("border-color", "#DCE5F5"),
                ],
            },
            {
                "selector": "td",
                "props": [
                    ("background-color", "#FFFFFF"),
                    ("color", "#24345B"),
                    ("border-color", "#E2E8F0"),
                ],
            },
        ])
    )

def hero(title, subtitle, icon="✨"):
    st.markdown(
        f"""
        <div class="hero">
          <h1>{icon} {title}</h1>
          <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def section_title(title, subtitle=None):
    st.markdown(f'<div class="section-chip">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.caption(subtitle)


def model_prediction_panel(report, student_id):
    """Collect the three unavailable inputs and run the saved Random Forest model."""
    st.divider()
    section_title(
        "تحلیل هوشمند با یادگیری ماشین",
        "سه ویژگی تکمیلی را وارد کنید؛ اطلاعات حضور و غیاب از پرونده دانش‌آموز خوانده می‌شود."
    )
    model = load_ml_model()

    if model is None:
        st.error(
            "فایل مدل پیدا نشد. ابتدا train_ml_model.py را اجرا کنید "
            "تا models/student_status_model.joblib ساخته شود."
        )
        return

    attendance = report["attendance"]

    with st.form(f"ml_prediction_form_{student_id}"):
        st.markdown("#### 🧠 اطلاعات تکمیلی برای پیش‌بینی")
        c1, c2, c3 = st.columns(3)

        with c1:
            study_hours = st.number_input(
                "میانگین ساعات مطالعه روزانه",
                min_value=0.0, max_value=16.0,
                value=4.0, step=0.5,
                help="میانگین تقریبی ساعات مطالعه روزانه دانش‌آموز"
            )
        with c2:
            assignment_rate = st.number_input(
                "درصد انجام تکالیف",
                min_value=0.0, max_value=100.0,
                value=80.0, step=1.0
            )
        with c3:
            previous_average = st.number_input(
                "میانگین نمرات دوره قبل",
                min_value=0.0, max_value=20.0,
                value=14.0, step=0.25
            )

        st.caption(
            f"داده‌های استخراج‌شده از پرونده: حضور {attendance['attendance_percentage']}٪، "
            f"غیبت {attendance['absent_days']} روز، تأخیر {attendance['late_days']} روز."
        )
        submitted = st.form_submit_button(
            "✨ تحلیل و پیش‌بینی وضعیت تحصیلی",
            use_container_width=True
        )

    if submitted:
        feature_values = pd.DataFrame([{
            "study_hours": float(study_hours),
            "attendance_percentage": float(attendance["attendance_percentage"]),
            "absence_days": int(attendance["absent_days"]),
            "late_days": int(attendance["late_days"]),
            "assignment_rate": float(assignment_rate),
            "previous_average": float(previous_average)
        }])

        expected_features = [
            "study_hours",
            "attendance_percentage",
            "absence_days",
            "late_days",
            "assignment_rate",
            "previous_average"
        ]

        try:
            feature_values = feature_values[expected_features]
            prediction = model.predict(feature_values)[0]
            probabilities = model.predict_proba(feature_values)[0]
            classes = list(model.classes_)

            confidence = float(np.max(probabilities)) * 100
            color = STATUS_COLORS.get(prediction, "#7657F6")

            st.markdown(
                f"""
                <div style="
                    background:linear-gradient(120deg,{color}18,rgba(255,255,255,.96));
                    border:1px solid {color}55;
                    border-right:7px solid {color};
                    border-radius:20px;padding:20px 22px;
                    box-shadow:0 8px 0 rgba(40,55,100,.05),0 15px 28px rgba(40,55,100,.08);
                    margin-top:16px;">
                    <div style="font-size: .95rem;color:#64748B;font-weight:700;">نتیجه پیش‌بینی مدل</div>
                    <div style="font-size:1.8rem;font-weight:900;color:{color};margin:5px 0;">{prediction}</div>
                    <div style="color:#475569;">بالاترین احتمال برآوردشده: {confidence:.1f}٪</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            probability_df = pd.DataFrame({
                "وضعیت تحصیلی": classes,
                "احتمال": probabilities * 100
            }).sort_values("احتمال", ascending=True)

            fig_prob = px.bar(
                probability_df,
                x="احتمال",
                y="وضعیت تحصیلی",
                orientation="h",
                text=probability_df["احتمال"].map(lambda x: f"{x:.1f}%"),
                color="وضعیت تحصیلی",
                color_discrete_map=STATUS_COLORS,
                title="احتمال برآوردشده برای هر وضعیت"
            )
            fig_prob.update_layout(
                template="plotly_white",
                font=dict(family="Vazirmatn", color="#263653", size=13),
                showlegend=False,
                xaxis_title="احتمال (درصد)",
                yaxis_title="",
                xaxis_range=[0, 100],
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=15, r=15, t=55, b=15)
            )
            fig_prob.update_traces(textposition="outside", cliponaxis=False)
            st.plotly_chart(fig_prob, use_container_width=True)

            st.info(
                "توجه: درصدهای بالا خروجی predict_proba مدل هستند و لزوماً "
                "احتمال‌های کالیبره‌شده یا تضمین‌شده نیستند. مدل فعلی با داده مصنوعی آموزش دیده است."
            )

        except Exception as exc:
            st.error(f"در اجرای پیش‌بینی خطایی رخ داد: {exc}")


# ==========================================================
# SIDEBAR
# ==========================================================
st.sidebar.markdown("## 🎓 پنل مدیریت")
st.sidebar.markdown("---")
st.sidebar.info("نسخه آزمایشی آموزشی\n\nاطلاعات این نسخه مصنوعی است.")
page = st.sidebar.radio(
    "بخش موردنظر را انتخاب کنید:",
    ["داشبورد اصلی", "پرونده دانش‌آموز", "پردازش تصویر و QR"]
)
st.sidebar.markdown("---")
st.sidebar.caption("نسخه هوشمند | طراحی رنگی و راست‌چین")


# ==========================================================
# STUDENT REPORT
# ==========================================================
def show_student_report(student_id):
    report = analyze_student(student_id)

    if report is None:
        st.error("پرونده دانش‌آموز پیدا نشد.")
        return

    student = report["student"]
    attendance = report["attendance"]

    st.markdown(
        f"## 👤 پرونده تحصیلی {student['first_name']} {student['last_name']}"
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("شناسه دانش‌آموز", student["student_id"])
    col2.metric("میانگین کل نمرات", f"{report['overall_average']} از ۲۰")
    col3.metric("درصد حضور", f"{attendance['attendance_percentage']}٪")

    st.info(
        f"وضعیت بر اساس قواعد فعلی: {report['academic_status']}  |  "
        f"وضعیت حضور و غیاب: {report['attendance_status']}"
    )

    section_title("مشخصات دانش‌آموز")
    info1, info2 = st.columns(2)
    with info1:
        st.write(f"**پایه:** {student['grade']}")
        st.write(f"**رشته:** {student['major']}")
    with info2:
        st.write(f"**کلاس:** {student['class_name']}")
        st.write(f"**نام:** {student['first_name']} {student['last_name']}")

    st.divider()
    section_title("عملکرد درسی", "مقایسه میانگین نمرات دانش‌آموز در درس‌های مختلف")
    subject_df = pd.DataFrame(report["subject_averages"])

    if not subject_df.empty:
        fig = px.bar(
            subject_df,
            x="subject_name",
            y="average_score",
            text="average_score",
            color="average_score",
            color_continuous_scale=["#F43F5E", "#F59E0B", "#10B981", "#2585F5"],
            range_y=[0, 20],
            labels={"subject_name": "نام درس", "average_score": "میانگین نمره"},
            title="مقایسه عملکرد دانش‌آموز در دروس"
        )
        fig.update_layout(
            template="plotly_white",
            font=dict(family="Vazirmatn", color="#263653", size=13),
            xaxis_title="درس",
            yaxis_title="نمره از ۲۰",
            coloraxis_showscale=False,
            plot_bgcolor="rgba(255,255,255,0)",
            paper_bgcolor="rgba(255,255,255,0)",
            xaxis=dict(tickfont=dict(color="#475569"), gridcolor="#E8EEF8"),
            yaxis=dict(tickfont=dict(color="#475569"), gridcolor="#E8EEF8")
        )
        fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

        display_subjects = subject_df.copy()
        display_subjects.columns = ["نام درس", "میانگین نمره"]
        st.dataframe(styled_table(display_subjects), use_container_width=True, hide_index=True)

    st.divider()
    section_title("وضعیت حضور و غیاب")
    a, b, c, d = st.columns(4)
    a.metric("کل روزها", attendance["total_days"])
    b.metric("حضور", attendance["present_days"])
    c.metric("غیبت", attendance["absent_days"])
    d.metric("تأخیر", attendance["late_days"])

    if report["grades"]:
        st.divider()
        section_title("جزئیات نمرات")
        grades_df = pd.DataFrame(report["grades"])
        grades_df.columns = ["نام درس", "نوبت", "نوع ارزیابی", "نمره", "تاریخ آزمون"]
        st.dataframe(styled_table(grades_df), use_container_width=True, hide_index=True)

    # ML prediction uses manual inputs for the three features not held in the current DB.
    model_prediction_panel(report, student_id)


# ==========================================================
# MAIN DASHBOARD
# ==========================================================
if page == "داشبورد اصلی":
    hero(
        "داشبورد تحلیلی مدیریت آموزشی",
        "نمای کلی وضعیت تحصیلی، نمرات و حضور و غیاب دانش‌آموزان",
        "📊"
    )

    connection = get_connection()
    try:
        cursor = connection.cursor()

        cursor.execute("SELECT COUNT(*) FROM students")
        total_students = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM grades")
        total_grades = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM attendance")
        total_attendance = cursor.fetchone()[0]

        cursor.execute("SELECT ROUND(AVG(score), 2) FROM grades")
        average_score = cursor.fetchone()[0] or 0

        cursor.execute("""
            SELECT s.subject_name, ROUND(AVG(g.score), 2) AS average_score
            FROM subjects s
            JOIN grades g ON s.subject_id = g.subject_id
            GROUP BY s.subject_id, s.subject_name
            ORDER BY average_score DESC
        """)
        subject_data = [dict(row) for row in cursor.fetchall()]

        cursor.execute("""
            SELECT status, COUNT(*) AS total
            FROM attendance
            GROUP BY status
        """)
        attendance_data = [dict(row) for row in cursor.fetchall()]

        cursor.execute("""
            WITH grade_stats AS (
                SELECT student_id, ROUND(AVG(score), 2) AS average_score
                FROM grades GROUP BY student_id
            ),
            attendance_stats AS (
                SELECT
                    student_id,
                    COUNT(*) AS total_days,
                    SUM(CASE WHEN status='present' THEN 1 ELSE 0 END) AS present_days,
                    SUM(CASE WHEN status='absent' THEN 1 ELSE 0 END) AS absent_days,
                    SUM(CASE WHEN status='late' THEN 1 ELSE 0 END) AS late_days
                FROM attendance GROUP BY student_id
            )
            SELECT
                s.student_id,
                s.first_name || ' ' || s.last_name AS student_name,
                COALESCE(g.average_score, 0) AS average_score,
                COALESCE(a.total_days, 0) AS total_days,
                COALESCE(a.present_days, 0) AS present_days,
                COALESCE(a.absent_days, 0) AS absent_days,
                COALESCE(a.late_days, 0) AS late_days,
                CASE WHEN COALESCE(a.total_days, 0)>0
                    THEN ROUND(100.0*a.present_days/a.total_days, 2)
                    ELSE 0 END AS attendance_percentage
            FROM students s
            LEFT JOIN grade_stats g ON s.student_id=g.student_id
            LEFT JOIN attendance_stats a ON s.student_id=a.student_id
        """)
        student_data = [dict(row) for row in cursor.fetchall()]
    finally:
        connection.close()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("👥 تعداد دانش‌آموزان", total_students)
    k2.metric("📝 نمرات ثبت‌شده", total_grades)
    k3.metric("⭐ میانگین کل کلاس", f"{average_score} از ۲۰")
    k4.metric("📅 سوابق حضور و غیاب", total_attendance)

    st.divider()
    left, right = st.columns([1.15, 0.85])

    with left:
        section_title("📚 عملکرد دروس")
        subject_df = pd.DataFrame(subject_data)
        if not subject_df.empty:
            fig_subject = px.bar(
                subject_df,
                x="subject_name",
                y="average_score",
                text="average_score",
                color="average_score",
                color_continuous_scale=["#F59E0B", "#10B981", "#2585F5"],
                range_y=[0, 20],
                labels={"subject_name": "نام درس", "average_score": "میانگین نمره"}
            )
            fig_subject.update_layout(
                template="plotly_white",
                font=dict(family="Vazirmatn", color="#263653", size=13),
                xaxis_title="نام درس",
                yaxis_title="نمره از ۲۰",
                coloraxis_showscale=False,
                plot_bgcolor="rgba(255,255,255,0)",
                paper_bgcolor="rgba(255,255,255,0)",
                xaxis=dict(tickfont=dict(color="#475569"), gridcolor="#E8EEF8"),
                yaxis=dict(tickfont=dict(color="#475569"), gridcolor="#E8EEF8")
            )
            fig_subject.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            st.plotly_chart(fig_subject, use_container_width=True)

            best = subject_df.iloc[0]
            weak = subject_df.iloc[-1]
            st.markdown(f"""<div class="status-success-card">قوی‌ترین درس: {best['subject_name']} با میانگین {best['average_score']}</div>""", unsafe_allow_html=True)
            st.markdown(f"""<div class="status-warning-card">کمترین میانگین: {weak['subject_name']} با میانگین {weak['average_score']}</div>""", unsafe_allow_html=True)

    with right:
        section_title("🗓️ توزیع حضور و غیاب")
        attendance_labels = {"present": "حاضر", "absent": "غایب", "late": "تأخیر"}
        attendance_df = pd.DataFrame(attendance_data)
        if not attendance_df.empty:
            attendance_df["status_label"] = attendance_df["status"].map(attendance_labels)
            fig_attendance = px.pie(
                attendance_df,
                names="status_label",
                values="total",
                hole=0.58,
                color="status_label",
                color_discrete_map={"حاضر": "#10B981", "غایب": "#F43F5E", "تأخیر": "#F59E0B"}
            )
            fig_attendance.update_layout(
                template="plotly_white",
                font=dict(family="Vazirmatn", color="#263653", size=13),
                legend_title_text="",
                plot_bgcolor="rgba(255,255,255,0)",
                paper_bgcolor="rgba(255,255,255,0)"
            )
            st.plotly_chart(fig_attendance, use_container_width=True)

    st.divider()
    left2, right2 = st.columns(2)

    student_df = pd.DataFrame(student_data)
    with left2:
        section_title("🏆 پنج دانش‌آموز برتر")
        ranking_df = student_df.sort_values("average_score", ascending=False).head(5)
        ranking_df = ranking_df[["student_id", "student_name", "average_score"]].copy()
        ranking_df.columns = ["شناسه", "نام دانش‌آموز", "میانگین نمرات"]
        st.dataframe(styled_table(ranking_df), use_container_width=True, hide_index=True)

    with right2:
        section_title("⚠️ دانش‌آموزان نیازمند توجه")
        attention_df = student_df[
            (student_df["average_score"] < 12) |
            (student_df["attendance_percentage"] < 75)
        ][["student_id", "student_name", "average_score", "attendance_percentage"]].copy()
        attention_df.columns = ["شناسه", "نام دانش‌آموز", "میانگین نمرات", "درصد حضور"]
        if not attention_df.empty:
            st.markdown(f"""<div class="status-warning-card">{len(attention_df)} دانش‌آموز بر اساس معیارهای اولیه نیازمند بررسی هستند.</div>""", unsafe_allow_html=True)
            st.dataframe(styled_table(attention_df), use_container_width=True, hide_index=True)
        else:
            st.markdown("""<div class="status-success-card">دانش‌آموزی بر اساس معیارهای فعلی نیازمند توجه ویژه تشخیص داده نشد.</div>""", unsafe_allow_html=True)

    st.caption("اطلاعات این نسخه آزمایشی مصنوعی است. نتایج مدل برای تصمیم‌گیری قطعی درباره دانش‌آموزان واقعی مناسب نیست.")


# ==========================================================
# STUDENT PROFILE
# ==========================================================
elif page == "پرونده دانش‌آموز":
    hero(
        "پرونده دانش‌آموز",
        "جست‌وجو، مشاهده سوابق آموزشی و دریافت تحلیل هوشمند",
        "👤"
    )
    with st.form("student_search_form"):
        student_id = st.text_input(
            "شناسه دانش‌آموز را وارد کنید",
            placeholder="برای مثال: 10001"
        )
        search_clicked = st.form_submit_button("🔎 نمایش پرونده", use_container_width=True)

    if search_clicked:
        if student_id.strip():
            show_student_report(student_id.strip())
        else:
            st.warning("لطفاً شناسه دانش‌آموز را وارد کنید.")


# ==========================================================
# IMAGE PROCESSING & QR
# ==========================================================
elif page == "پردازش تصویر و QR":
    hero(
        "پردازش تصویر و شناسایی دانش‌آموز",
        "تشخیص چهره از تصویر و بازیابی پرونده با QR Code",
        "📷"
    )

    face_tab, qr_tab = st.tabs(["🙂 تشخیص چهره", "🔳 خواندن QR Code"])

    with face_tab:
        section_title("تشخیص چهره از تصویر")
        face_source = st.radio(
            "روش دریافت تصویر:",
            ["بارگذاری تصویر", "استفاده از دوربین"],
            horizontal=True
        )
        uploaded_image = None

        if face_source == "بارگذاری تصویر":
            uploaded_image = st.file_uploader(
                "تصویر را انتخاب کنید:",
                type=["jpg", "jpeg", "png"],
                key="face_upload"
            )
        else:
            uploaded_image = st.camera_input("تصویر را با دوربین ثبت کنید:")

        if uploaded_image is not None:
            pil_image = Image.open(uploaded_image).convert("RGB")
            image_array = np.array(pil_image)
            image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            result = detect_faces(image_bgr)

            st.image(
                cv2.cvtColor(result["image"], cv2.COLOR_BGR2RGB),
                caption="نتیجه تشخیص چهره",
                use_container_width=True
            )
            st.metric("تعداد چهره‌های تشخیص داده‌شده", result["face_count"])

            if result["face_count"] > 0:
                st.success("چهره در تصویر تشخیص داده شد.")
            else:
                st.warning("هیچ چهره‌ای در تصویر پیدا نشد.")

            st.caption("تشخیص چهره به معنی شناسایی هویت فرد نیست.")

    with qr_tab:
        section_title("خواندن QR Code دانش‌آموز")
        qr_file = st.file_uploader(
            "تصویر QR Code را بارگذاری کنید:",
            type=["png", "jpg", "jpeg"],
            key="qr_upload"
        )

        if qr_file is not None:
            qr_pil = Image.open(qr_file).convert("RGB")
            qr_array = np.array(qr_pil)
            qr_bgr = cv2.cvtColor(qr_array, cv2.COLOR_RGB2BGR)
            token = read_qr_code(qr_bgr)

            if token:
                st.success("QR Code با موفقیت خوانده شد.")
                student = get_student_by_qr(token)

                if student:
                    st.markdown("### 👤 اطلاعات دانش‌آموز")
                    q1, q2, q3 = st.columns(3)
                    q1.metric("شناسه", student["student_id"])
                    q2.metric("پایه", student["grade"])
                    q3.metric("کلاس", student["class_name"])
                    st.write(f"**نام:** {student['first_name']} {student['last_name']}")
                    st.write(f"**رشته:** {student['major']}")
                    st.divider()
                    show_student_report(student["student_id"])
                else:
                    st.error("برای این QR Code پرونده‌ای پیدا نشد.")
            else:
                st.error("QR Code قابل شناسایی نیست. لطفاً تصویر واضح‌تری انتخاب کنید.")


# ==========================================================
# FOOTER
# ==========================================================
st.divider()
st.markdown(
    """
    <div style="text-align:center;color:#718096;padding:8px 0 2px 0;font-size:.85rem;">
      🎓 سامانه هوشمند دانش‌آموزی | نسخه آزمایشی آموزشی
      <br>طراحی رنگی • تحلیل داده • یادگیری ماشین
    </div>
    """,
    unsafe_allow_html=True
)
