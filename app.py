import sqlite3
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import cv2
import numpy as np
import joblib
from modules.student_registration import register_student
from modules.face_recognition import FaceRecognition

from PIL import Image

from modules.academic_analysis import analyze_student
from modules.qr_manager import read_qr_code, get_student_by_qr
from modules.face_detection import detect_faces
from modules.database_manager import (
    get_connection,
    create_tables,
    get_all_face_templates
)
from auth_manager import (initialize_auth_table, count_users, create_user, authenticate_user, list_users, set_user_active, change_user_role, reset_user_password, generate_temporary_password, change_own_password, initialize_access_tables, get_teacher_assignments, set_teacher_assignments, user_can_access_student, get_accessible_students, get_accessible_subjects)
#from modules.database_manager import get_connection, create_tables

from data.generate_data import (
    generate_students,
    generate_subjects,
    generate_grades,
    generate_attendance
)

# ==========================================================
# PDF REPORT HELPER FOR CLASS / SCHOOL ANALYTICS
# ==========================================================
def build_class_analytics_pdf(summary_df, subject_df, status_df, metrics, filters):
    """Create a Persian PDF report with KPIs, charts and student summary."""
    import io
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_RIGHT, TA_CENTER
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        Image as RLImage, PageBreak, KeepTogether
    )

    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
    except ImportError as exc:
        raise ImportError(
            "برای نمایش صحیح متن فارسی در PDF، بسته‌های arabic-reshaper و python-bidi لازم هستند."
        ) from exc

    def rtl(value):
        value = str(value)
        return get_display(arabic_reshaper.reshape(value))

    # Use a Persian-capable font for BOTH ReportLab and Matplotlib.
    # Matplotlib must receive the font file explicitly; setting only a generic
    # family name can cause fallback fonts and broken Persian glyph shaping.
    from matplotlib.font_manager import FontProperties

    windows_fonts = os.path.join(os.environ.get("WINDIR", "C:/Windows"), "Fonts")
    font_candidates = [
        os.path.join(windows_fonts, "arial.ttf"),
        os.path.join(windows_fonts, "tahoma.ttf"),
        "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoKufiArabic-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansArabic-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    font_path = next((f for f in font_candidates if os.path.exists(f)), None)
    if font_path:
        pdfmetrics.registerFont(TTFont("PersianFont", font_path))
        base_font = "PersianFont"
        chart_font = FontProperties(fname=font_path)
        plt.rcParams["font.family"] = chart_font.get_name()
        plt.rcParams["axes.unicode_minus"] = False
    else:
        base_font = "Helvetica"
        chart_font = FontProperties(family="DejaVu Sans")

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=landscape(A4), rightMargin=13*mm,
        leftMargin=13*mm, topMargin=12*mm, bottomMargin=12*mm
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "PersianTitle", parent=styles["Title"], fontName=base_font,
        fontSize=18, leading=25, alignment=TA_CENTER, textColor=colors.HexColor("#24345B")
    )
    body_style = ParagraphStyle(
        "PersianBody", parent=styles["BodyText"], fontName=base_font,
        fontSize=9, leading=15, alignment=TA_RIGHT
    )
    heading_style = ParagraphStyle(
        "PersianHeading", parent=styles["Heading2"], fontName=base_font,
        fontSize=12, leading=18, alignment=TA_RIGHT, textColor=colors.HexColor("#315B91")
    )

    story = [Paragraph(rtl("گزارش تحلیلی عملکرد کلاس و مدرسه"), title_style), Spacer(1, 5*mm)]
    filter_line = " | ".join([f"{k}: {v}" for k, v in filters.items()])
    story.append(Paragraph(rtl("فیلترهای گزارش: " + filter_line), body_style))
    story.append(Spacer(1, 4*mm))

    metric_rows = [
        [rtl("تعداد دانش‌آموزان"), rtl("میانگین نمرات"), rtl("درصد حضور"), rtl("تعداد غیبت")],
        [str(metrics["students"]), f'{metrics["average"]:.2f} / 20', f'{metrics["attendance"]:.1f}٪', str(metrics["absent"])],
    ]
    mt = Table(metric_rows, colWidths=[65*mm]*4, rowHeights=[10*mm, 12*mm])
    mt.setStyle(TableStyle([
        ("FONTNAME", (0,0), (-1,-1), base_font), ("FONTSIZE", (0,0), (-1,-1), 10),
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#EAF1FB")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.HexColor("#24345B")),
        ("ALIGN", (0,0), (-1,-1), "CENTER"), ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("GRID", (0,0), (-1,-1), 0.4, colors.HexColor("#CCD6E5")),
    ]))
    story += [mt, Spacer(1, 6*mm)]

    # IMPORTANT:
    # Matplotlib does not consistently shape Persian/Arabic text across systems.
    # Therefore chart images contain numeric labels only. All Persian titles,
    # legends and category names are rendered by ReportLab, which is already
    # displaying Persian correctly in the report tables and headings.
    chart_sections = []

    if not subject_df.empty:
        fig, ax = plt.subplots(figsize=(8.5, 3.4))
        plot_df = subject_df.sort_values("average_score", ascending=True).reset_index(drop=True)
        y_positions = list(range(len(plot_df)))
        ax.barh(y_positions, plot_df["average_score"])
        ax.set_yticks(y_positions)
        ax.set_yticklabels([str(i + 1) for i in y_positions])
        ax.set_xlim(0, 20)
        ax.grid(axis="x", alpha=.2)
        fig.tight_layout()
        img = io.BytesIO()
        fig.savefig(img, format="png", dpi=180, bbox_inches="tight")
        plt.close(fig)
        img.seek(0)

        subject_legend = [[rtl("ردیف"), rtl("نام درس"), rtl("میانگین نمره")]]
        for i, row in plot_df.iterrows():
            subject_legend.append([
                str(i + 1),
                rtl(row["subject_name"]),
                f'{float(row["average_score"]):.2f}'
            ])
        subject_table = Table(subject_legend, colWidths=[20*mm, 95*mm, 35*mm])
        subject_table.setStyle(TableStyle([
            ("FONTNAME", (0,0), (-1,-1), base_font),
            ("FONTSIZE", (0,0), (-1,-1), 8),
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#EAF1FB")),
            ("ALIGN", (0,0), (-1,-1), "CENTER"),
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
            ("GRID", (0,0), (-1,-1), .35, colors.HexColor("#D5DCE8")),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F7F9FC")]),
        ]))
        chart_sections.append((
            rtl("میانگین نمرات درس‌ها"),
            img,
            subject_table
        ))

    if not status_df.empty:
        fig, ax = plt.subplots(figsize=(5.4, 3.6))
        ax.pie(
            status_df["count"],
            labels=None,
            autopct="%1.1f%%",
            startangle=90
        )
        fig.tight_layout()
        img = io.BytesIO()
        fig.savefig(img, format="png", dpi=180, bbox_inches="tight")
        plt.close(fig)
        img.seek(0)

        status_legend = [[rtl("وضعیت"), rtl("تعداد")]]
        for _, row in status_df.iterrows():
            status_legend.append([
                rtl(row["status_label"]),
                str(int(row["count"]))
            ])
        status_table = Table(status_legend, colWidths=[75*mm, 35*mm])
        status_table.setStyle(TableStyle([
            ("FONTNAME", (0,0), (-1,-1), base_font),
            ("FONTSIZE", (0,0), (-1,-1), 8),
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#EAF1FB")),
            ("ALIGN", (0,0), (-1,-1), "CENTER"),
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
            ("GRID", (0,0), (-1,-1), .35, colors.HexColor("#D5DCE8")),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F7F9FC")]),
        ]))
        chart_sections.append((
            rtl("توزیع وضعیت حضور و غیاب"),
            img,
            status_table
        ))

    if chart_sections:
        story.append(Paragraph(rtl("نمودارهای تحلیلی"), heading_style))
        for chart_title, chart_img, chart_legend in chart_sections:
            story.append(Paragraph(chart_title, body_style))
            story.append(RLImage(chart_img, width=125*mm, height=68*mm, kind="proportional"))
            story.append(chart_legend)
            story.append(Spacer(1, 4*mm))

    story.append(Paragraph(rtl("خلاصه عملکرد دانش‌آموزان"), heading_style))
    cols = ["student_id", "first_name", "last_name", "class_name", "average_score", "attendance_percentage", "absent_days"]
    headers = ["شناسه", "نام", "نام خانوادگی", "کلاس", "میانگین نمرات", "درصد حضور", "تعداد غیبت"]
    table_data = [[rtl(x) for x in headers]]
    for _, row in summary_df.head(35).iterrows():
        vals = [row.get(c, "") for c in cols]
        vals[4] = f'{float(vals[4]):.2f}' if pd.notna(vals[4]) else "—"
        vals[5] = f'{float(vals[5]):.1f}٪' if pd.notna(vals[5]) else "—"
        vals[6] = str(int(vals[6])) if pd.notna(vals[6]) else "0"
        table_data.append([rtl(v) for v in vals])
    student_table = Table(table_data, repeatRows=1, colWidths=[25*mm, 24*mm, 32*mm, 32*mm, 30*mm, 30*mm, 25*mm])
    student_table.setStyle(TableStyle([
        ("FONTNAME", (0,0), (-1,-1), base_font), ("FONTSIZE", (0,0), (-1,-1), 7.5),
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#EAF1FB")),
        ("ALIGN", (0,0), (-1,-1), "CENTER"), ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("GRID", (0,0), (-1,-1), .35, colors.HexColor("#D5DCE8")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#F7F9FC")]),
    ]))
    story.append(student_table)
    story.append(Spacer(1, 4*mm))
    story.append(Paragraph(rtl("توضیح: این گزارش بر اساس داده‌های ثبت‌شده و قواعد آماری اولیه تهیه شده است. نتایج هشداردهنده به معنی تشخیص قطعی نیستند و بررسی انسانی هنرآموز یا مدیر ضروری است."), body_style))
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


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

# ==========================================================
# AUTHENTICATION GATE
# ==========================================================
initialize_auth_table()
initialize_access_tables()

if "authenticated_user" not in st.session_state:
    st.session_state.authenticated_user = None

if st.session_state.authenticated_user is None:
    st.markdown("<div style='height:35px'></div>", unsafe_allow_html=True)
    st.markdown("# 🎓 سامانه هوشمند دانش‌آموزی")
    st.caption("ورود امن کاربران سامانه")

    if count_users() == 0:
        st.info("برای راه‌اندازی اولیه، حساب مدیر سامانه را ایجاد کنید. این فرم فقط تا زمان ایجاد اولین حساب نمایش داده می‌شود.")
        with st.form("first_admin_setup", clear_on_submit=False):
            full_name = st.text_input("نام و نام خانوادگی مدیر")
            username = st.text_input("نام کاربری مدیر")
            password = st.text_input("رمز عبور (حداقل ۱۰ نویسه)", type="password")
            password_confirm = st.text_input("تکرار رمز عبور", type="password")
            create_admin = st.form_submit_button("ایجاد حساب مدیر", use_container_width=True)

        if create_admin:
            if password != password_confirm:
                st.error("رمز عبور و تکرار آن یکسان نیستند.")
            else:
                try:
                    create_user(username, full_name, password, role="admin")
                    st.success("حساب مدیر ساخته شد. اکنون با نام کاربری و رمز عبور وارد شوید.")
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
                except Exception as exc:
                    if "UNIQUE" in str(exc).upper():
                        st.error("این نام کاربری قبلاً ثبت شده است.")
                    else:
                        st.error(f"ایجاد حساب با خطا مواجه شد: {exc}")
    else:
        with st.form("login_form", clear_on_submit=True):
            username = st.text_input("نام کاربری")
            password = st.text_input("رمز عبور", type="password")
            login_clicked = st.form_submit_button("ورود به سامانه", use_container_width=True)

        if login_clicked:
            user = authenticate_user(username, password)
            if user:
                st.session_state.authenticated_user = user
                st.rerun()
            else:
                st.error("نام کاربری یا رمز عبور نادرست است، یا حساب غیرفعال شده است.")

    st.stop()

# Force users with temporary/reset credentials to choose a personal password.
if st.session_state.authenticated_user and st.session_state.authenticated_user.get("must_change_password"):
    forced_user = st.session_state.authenticated_user
    st.markdown("# 🔐 تغییر اجباری رمز عبور")
    st.warning("رمز عبور شما موقت است. برای ادامه کار، یک رمز شخصی جدید تعیین کنید.")
    with st.form("mandatory_password_change"):
        own_new_password = st.text_input("رمز عبور جدید (حداقل ۱۰ نویسه)", type="password")
        own_new_password_confirm = st.text_input("تکرار رمز عبور جدید", type="password")
        own_change_submit = st.form_submit_button("ثبت رمز جدید", use_container_width=True)
    if own_change_submit:
        if own_new_password != own_new_password_confirm:
            st.error("رمز عبور و تکرار آن یکسان نیستند.")
        else:
            try:
                change_own_password(forced_user["user_id"], own_new_password)
                st.session_state.authenticated_user["must_change_password"] = False
                st.success("رمز عبور تغییر کرد. اکنون وارد سامانه می‌شوید.")
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
    st.stop()


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

/* رنگ اختصاصی پیام چهره ناشناس */
section.main .unknown-face-message,
section.main .unknown-face-message * {
    color: #FFD700 !important;
    -webkit-text-fill-color: #FFD700 !important;
}

section.main .unknown-face-message {
    background-color: #393939 !important;
    padding: 15px 20px;
    border-radius: 10px;
    border: 1px solid #FFD700;
    font-size: 20px;
    font-weight: 900;
    text-align: right;
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
    st.markdown(
        f'<div class="section-chip">{title}</div>',
        unsafe_allow_html=True
    )
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

            fig_prob.update_traces(
                textposition="outside",
                cliponaxis=False
            )

            st.plotly_chart(
                fig_prob,
                use_container_width=True
            )


            st.info(
                "توجه: درصدهای بالا خروجی predict_proba مدل هستند و لزوماً "
                "احتمال‌های کالیبره‌شده یا تضمین‌شده نیستند. مدل فعلی با داده مصنوعی آموزش دیده است."
            )

        except Exception as exc:
            st.error(f"در اجرای پیش‌بینی خطایی رخ داد: {exc}")



def go_to_registration(photo_bytes):
    st.session_state["unknown_face_photo"] = photo_bytes
    st.session_state["page_selector"] = "ثبت دانش‌آموز"

# ==========================================================
# DISPLAY STUDENT PHOTO
# ==========================================================

def display_student_photo(photo_path):
    """Display the student's saved photo."""

    if not photo_path:
        st.warning("تصویری برای این دانش‌آموز ثبت نشده است.")
        return

    image_path = Path(str(photo_path))

    # Resolve relative paths from project directory
    if not image_path.is_absolute():
        image_path = BASE_DIR / image_path

    if image_path.exists() and image_path.is_file():
        st.image(
            str(image_path),
            caption="تصویر دانش‌آموز",
            width=250
        )
    else:
        st.warning("فایل تصویر دانش‌آموز در مسیر ذخیره‌شده پیدا نشد.")
        st.caption(f"مسیر ثبت‌شده: {photo_path}")

# ==========================================================
# SIDEBAR
# ==========================================================
st.sidebar.markdown("## 🎓 پنل مدیریت")
current_user = st.session_state.authenticated_user
st.sidebar.success(f"کاربر: {current_user['full_name']}")
role_labels = {"admin": "مدیر سامانه", "deputy": "معاون آموزشی", "teacher": "هنرآموز"}
st.sidebar.caption(f"سطح حساب: {role_labels.get(current_user['role'], 'کاربر')}")
if st.sidebar.button("خروج از حساب", use_container_width=True):
    st.session_state.authenticated_user = None
    st.rerun()
st.sidebar.markdown("---")
st.sidebar.info("نسخه آزمایشی آموزشی\n\nاطلاعات این نسخه مصنوعی است.")

if current_user["role"] == "teacher":
    menu_items = ["مدیریت نمرات", "مدیریت حضور و غیاب"]
elif current_user["role"] == "deputy":
    menu_items = [
        "داشبورد اصلی", "پرونده دانش‌آموز", "ثبت دانش‌آموز",
        "مدیریت نمرات", "مدیریت حضور و غیاب",
        "تحلیل هوشمند کلاس و مدرسه", "پردازش تصویر و QR"
    ]
else:
    menu_items = [
        "داشبورد اصلی", "پرونده دانش‌آموز", "ثبت دانش‌آموز",
        "مدیریت نمرات", "مدیریت حضور و غیاب",
        "تحلیل هوشمند کلاس و مدرسه", "پردازش تصویر و QR", "مدیریت کاربران"
    ]

page = st.sidebar.radio(
    "بخش موردنظر را انتخاب کنید:",
    menu_items,
    key="page_selector"
)

if current_user["role"] == "teacher" and page not in {
    "مدیریت نمرات", "مدیریت حضور و غیاب"
}:
    st.error("این نقش اجازه دسترسی به بخش انتخاب‌شده را ندارد.")
    st.stop()

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

    # ==========================================================
    # AI ACADEMIC PERFORMANCE ANALYSIS
    # ==========================================================
    st.divider()
    section_title(
        "تحلیل هوشمند عملکرد تحصیلی",
        "بررسی نقاط قوت، نقاط نیازمند تقویت و روند تغییر نمرات"
    )

    performance = report["performance_analysis"]

    # Academic strengths and weaknesses
    strong_subjects = performance["strong_subjects"]
    weak_subjects = performance["weak_subjects"]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### نقاط قوت")
        if strong_subjects:
            for subject in strong_subjects:
                st.success(f"⭐ {subject}")
        else:
            st.info("هنوز درس قوی مشخص نشده است.")

    with col2:
        st.markdown("### درس‌های نیازمند تقویت")
        if weak_subjects:
            for subject in weak_subjects:
                st.warning(f"📌 {subject}")
        else:
            st.info("در حال حاضر درس ضعیفی شناسایی نشده است.")

    # Term averages and academic trend
    st.markdown("### روند تغییر میانگین نمرات")

    term_df = pd.DataFrame(performance["term_averages"])

    if not term_df.empty:
        fig = px.line(
            term_df,
            x="term",
            y="average_score",
            markers=True,
            text="average_score",
            range_y=[0, 20],
            labels={
                "term": "نوبت تحصیلی",
                "average_score": "میانگین نمره"
            },
            title="روند میانگین نمرات در نوبت‌های تحصیلی"
        )

        fig.update_traces(
            line=dict(width=3),
            marker=dict(size=10),
            texttemplate="%{text:.2f}",
            textposition="top center"
        )

        fig.update_layout(
            template="plotly_white",
            font=dict(family="Vazirmatn", color="#263653", size=13),
            plot_bgcolor="rgba(255,255,255,0)",
            paper_bgcolor="rgba(255,255,255,0)",
            yaxis=dict(range=[0, 20], title="نمره از ۲۰"),
            xaxis_title="نوبت تحصیلی"
        )

        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("برای رسم نمودار، ابتدا نمرات نوبت‌های تحصیلی را ثبت کنید.")

    # Trend status
    st.markdown("### وضعیت روند تحصیلی")

    trend = performance["trend"]
    trend_change = performance["trend_change"]

    if trend == "روند پیشرفت تحصیلی":
        st.success(f"📈 {trend}")
    elif trend == "روند افت تحصیلی":
        st.warning(f"📉 {trend}")
    else:
        st.info(f"ℹ️ {trend}")

    if trend_change is not None:
        st.metric(
            "میزان تغییر میانگین",
            f"{trend_change:+.2f} نمره"
        )

    # Educational recommendations
    st.markdown("### پیشنهادهای آموزشی")

    for recommendation in performance["recommendations"]:
        st.info(f"💡 {recommendation}")


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
# CLASS / SCHOOL ANALYTICS DASHBOARD
# ==========================================================
elif page == "تحلیل هوشمند کلاس و مدرسه":

    hero(
        "تحلیل هوشمند کلاس و مدرسه",
        "بررسی تجمیعی عملکرد درسی و وضعیت حضور و غیاب بر اساس اطلاعات ثبت‌شده",
        "📊"
    )

    connection = get_connection()
    try:
        students_df = pd.read_sql_query("""
            SELECT student_id, first_name, last_name, grade, major, class_name
            FROM students
            ORDER BY grade, major, class_name, last_name, first_name
        """, connection)

        grades_df = pd.read_sql_query("""
            SELECT
                g.student_id,
                s.grade,
                s.major,
                s.class_name,
                sub.subject_name,
                g.term,
                g.score
            FROM grades g
            JOIN students s ON s.student_id = g.student_id
            JOIN subjects sub ON sub.subject_id = g.subject_id
        """, connection)

        attendance_df = pd.read_sql_query("""
            SELECT
                a.student_id,
                s.grade,
                s.major,
                s.class_name,
                a.status
            FROM attendance a
            JOIN students s ON s.student_id = a.student_id
        """, connection)
    finally:
        connection.close()

    if students_df.empty:
        st.warning("هنوز دانش‌آموزی برای تحلیل در پایگاه داده ثبت نشده است.")
    else:
        # -------------------- FILTERS --------------------
        section_title("فیلتر اطلاعات", "برای مشاهده گزارش کل مدرسه، همه گزینه‌ها را روی «همه» قرار دهید.")
        f1, f2, f3 = st.columns(3)

        grade_options = ["همه"] + sorted(
            students_df["grade"].dropna().astype(str).unique().tolist()
        )
        with f1:
            selected_grade = st.selectbox(
                "پایه تحصیلی", grade_options, key="analytics_grade_filter"
            )

        grade_scope = students_df.copy()
        if selected_grade != "همه":
            grade_scope = grade_scope[
                grade_scope["grade"].astype(str) == selected_grade
            ]

        major_options = ["همه"] + sorted(
            grade_scope["major"].dropna().astype(str).unique().tolist()
        )
        with f2:
            selected_major = st.selectbox(
                "رشته", major_options, key="analytics_major_filter"
            )

        major_scope = grade_scope.copy()
        if selected_major != "همه":
            major_scope = major_scope[
                major_scope["major"].astype(str) == selected_major
            ]

        class_options = ["همه"] + sorted(
            major_scope["class_name"].dropna().astype(str).unique().tolist()
        )
        with f3:
            selected_class = st.selectbox(
                "کلاس", class_options, key="analytics_class_filter"
            )

        filtered_students = major_scope.copy()
        if selected_class != "همه":
            filtered_students = filtered_students[
                filtered_students["class_name"].astype(str) == selected_class
            ]

        selected_ids = filtered_students["student_id"].astype(str).tolist()
        filtered_grades = grades_df[
            grades_df["student_id"].astype(str).isin(selected_ids)
        ].copy()
        filtered_attendance = attendance_df[
            attendance_df["student_id"].astype(str).isin(selected_ids)
        ].copy()

        if filtered_students.empty:
            st.info("برای ترکیب فیلترهای انتخاب‌شده دانش‌آموزی پیدا نشد.")
        else:
            # -------------------- STUDENT-LEVEL SUMMARY --------------------
            if not filtered_grades.empty:
                per_student_grades = (
                    filtered_grades.groupby("student_id", as_index=False)["score"]
                    .mean()
                    .rename(columns={"score": "average_score"})
                )
            else:
                per_student_grades = pd.DataFrame(
                    columns=["student_id", "average_score"]
                )

            if not filtered_attendance.empty:
                attendance_summary = (
                    filtered_attendance.assign(
                        is_present=(filtered_attendance["status"] == "present").astype(int),
                        is_absent=(filtered_attendance["status"] == "absent").astype(int),
                        is_late=(filtered_attendance["status"] == "late").astype(int),
                    )
                    .groupby("student_id", as_index=False)
                    .agg(
                        total_days=("status", "count"),
                        present_days=("is_present", "sum"),
                        absent_days=("is_absent", "sum"),
                        late_days=("is_late", "sum"),
                    )
                )
                attendance_summary["attendance_percentage"] = (
                    100 * attendance_summary["present_days"]
                    / attendance_summary["total_days"].replace(0, 1)
                ).round(1)
            else:
                attendance_summary = pd.DataFrame(columns=[
                    "student_id", "total_days", "present_days",
                    "absent_days", "late_days", "attendance_percentage"
                ])

            student_summary = (
                filtered_students.copy()
                .merge(per_student_grades, on="student_id", how="left")
                .merge(attendance_summary, on="student_id", how="left")
            )
            numeric_defaults = {
                "average_score": 0,
                "total_days": 0,
                "present_days": 0,
                "absent_days": 0,
                "late_days": 0,
                "attendance_percentage": 0,
            }
            for col, default in numeric_defaults.items():
                student_summary[col] = student_summary[col].fillna(default)

            student_summary["academic_status"] = student_summary[
                "average_score"
            ].apply(
                lambda x: (
                    "عالی" if x >= 17 else
                    "خوب" if x >= 14 else
                    "متوسط" if x >= 12 else
                    "نیازمند بررسی"
                )
            )

            total_students = len(filtered_students)
            average_score = (
                round(float(filtered_grades["score"].mean()), 2)
                if not filtered_grades.empty else 0
            )
            total_attendance = len(filtered_attendance)
            present_count = int((filtered_attendance["status"] == "present").sum())
            absent_count = int((filtered_attendance["status"] == "absent").sum())
            late_count = int((filtered_attendance["status"] == "late").sum())
            attendance_rate = (
                round(100 * present_count / total_attendance, 1)
                if total_attendance else 0
            )

            st.divider()
            section_title("شاخص‌های کلی", "مقادیر بر اساس فیلترهای انتخاب‌شده محاسبه شده‌اند.")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("👥 تعداد دانش‌آموزان", total_students)
            k2.metric("📝 میانگین نمرات ثبت‌شده", f"{average_score} از ۲۰")
            k3.metric("📅 درصد حضور", f"{attendance_rate}٪")
            k4.metric("⚠️ تعداد غیبت‌ها", absent_count)

            # -------------------- CHARTS --------------------
            st.divider()
            chart_left, chart_right = st.columns(2)

            with chart_left:
                section_title("میانگین نمرات درس‌ها")
                if filtered_grades.empty:
                    st.info("برای این محدوده هنوز نمره‌ای ثبت نشده است.")
                else:
                    subject_summary = (
                        filtered_grades.groupby("subject_name", as_index=False)["score"]
                        .mean()
                        .rename(columns={"score": "average_score"})
                        .sort_values("average_score", ascending=False)
                    )
                    fig_subjects = px.bar(
                        subject_summary,
                        x="subject_name",
                        y="average_score",
                        text="average_score",
                        color="average_score",
                        color_continuous_scale=["#F43F5E", "#F59E0B", "#10B981", "#2585F5"],
                        range_y=[0, 20],
                        labels={
                            "subject_name": "نام درس",
                            "average_score": "میانگین نمره"
                        },
                        title="مقایسه میانگین نمرات دروس"
                    )
                    fig_subjects.update_traces(
                        texttemplate="%{text:.2f}", textposition="outside"
                    )
                    fig_subjects.update_layout(
                        template="plotly_white",
                        font=dict(family="Vazirmatn", color="#263653", size=12),
                        xaxis_title="درس",
                        yaxis_title="نمره از ۲۰",
                        coloraxis_showscale=False,
                        plot_bgcolor="rgba(255,255,255,0)",
                        paper_bgcolor="rgba(255,255,255,0)"
                    )
                    st.plotly_chart(fig_subjects, use_container_width=True)

            with chart_right:
                section_title("توزیع وضعیت حضور و غیاب")
                if filtered_attendance.empty:
                    st.info("برای این محدوده سابقه حضور و غیابی ثبت نشده است.")
                else:
                    status_labels = {
                        "present": "حاضر",
                        "absent": "غایب",
                        "late": "تأخیر"
                    }
                    status_summary = (
                        filtered_attendance["status"].map(status_labels)
                        .fillna("نامشخص").value_counts()
                        .rename_axis("وضعیت").reset_index(name="تعداد")
                    )
                    fig_attendance = px.pie(
                        status_summary,
                        names="وضعیت",
                        values="تعداد",
                        hole=0.55,
                        color="وضعیت",
                        color_discrete_map={
                            "حاضر": "#10B981",
                            "غایب": "#F43F5E",
                            "تأخیر": "#F59E0B",
                            "نامشخص": "#94A3B8"
                        },
                        title="توزیع رکوردهای حضور و غیاب"
                    )
                    fig_attendance.update_layout(
                        template="plotly_white",
                        font=dict(family="Vazirmatn", color="#263653", size=12),
                        legend_title_text="",
                        plot_bgcolor="rgba(255,255,255,0)",
                        paper_bgcolor="rgba(255,255,255,0)"
                    )
                    st.plotly_chart(fig_attendance, use_container_width=True)

            # Compare classes using separate scales to avoid misleading comparisons.
            if selected_class == "همه" and student_summary["class_name"].nunique() > 1:
                st.divider()
                section_title("مقایسه کلاس‌ها", "برای جلوگیری از مقایسه مقیاس‌های متفاوت، دو نمودار جداگانه نمایش داده می‌شود.")
                class_summary = (
                    student_summary.groupby("class_name", as_index=False)
                    .agg(
                        average_score=("average_score", "mean"),
                        attendance_percentage=("attendance_percentage", "mean"),
                        students_count=("student_id", "count")
                    )
                )
                class_summary["average_score"] = class_summary["average_score"].round(2)
                class_summary["attendance_percentage"] = class_summary["attendance_percentage"].round(1)
                score_col, attendance_col = st.columns(2)
                with score_col:
                    section_title("میانگین نمرات کلاس‌ها")
                    fig_class_scores = px.bar(
                        class_summary.sort_values("average_score", ascending=False),
                        x="class_name", y="average_score", text="average_score",
                        range_y=[0, 20], labels={"class_name": "کلاس", "average_score": "میانگین نمره از ۲۰"},
                        title="مقایسه میانگین نمرات کلاس‌ها"
                    )
                    fig_class_scores.update_traces(texttemplate="%{text:.2f}", textposition="outside")
                    fig_class_scores.update_layout(template="plotly_white", font=dict(family="Vazirmatn", color="#263653", size=11), yaxis_title="نمره از ۲۰", xaxis_title="کلاس", plot_bgcolor="rgba(255,255,255,0)", paper_bgcolor="rgba(255,255,255,0)")
                    st.plotly_chart(fig_class_scores, use_container_width=True)
                with attendance_col:
                    section_title("درصد حضور کلاس‌ها")
                    fig_class_attendance = px.bar(
                        class_summary.sort_values("attendance_percentage", ascending=False),
                        x="class_name", y="attendance_percentage", text="attendance_percentage",
                        range_y=[0, 100], labels={"class_name": "کلاس", "attendance_percentage": "درصد حضور"},
                        title="مقایسه درصد حضور کلاس‌ها"
                    )
                    fig_class_attendance.update_traces(texttemplate="%{text:.1f}٪", textposition="outside")
                    fig_class_attendance.update_layout(template="plotly_white", font=dict(family="Vazirmatn", color="#263653", size=11), yaxis_title="درصد حضور", xaxis_title="کلاس", plot_bgcolor="rgba(255,255,255,0)", paper_bgcolor="rgba(255,255,255,0)")
                    st.plotly_chart(fig_class_attendance, use_container_width=True)

            # -------------------- ATTENTION LIST --------------------
            st.divider()
            section_title(
                "دانش‌آموزان نیازمند بررسی",
                "این فهرست بر اساس آستانه‌های ساده گزارش‌گیری ساخته شده است، نه تشخیص قطعی."
            )
            attention_df = student_summary[
                (
                    (student_summary["average_score"] > 0)
                    & (student_summary["average_score"] < 12)
                )
                | (
                    (student_summary["total_days"] > 0)
                    & (student_summary["attendance_percentage"] < 75)
                )
            ][[
                "student_id", "first_name", "last_name", "class_name",
                "average_score", "attendance_percentage", "absent_days"
            ]].copy()
            attention_df.columns = [
                "شناسه", "نام", "نام خانوادگی", "کلاس",
                "میانگین نمرات", "درصد حضور", "تعداد غیبت"
            ]
            if not attention_df.empty:
                attention_df["میانگین نمرات"] = pd.to_numeric(attention_df["میانگین نمرات"], errors="coerce").round(2)
                attention_df["درصد حضور"] = pd.to_numeric(attention_df["درصد حضور"], errors="coerce").round(1)

            if attention_df.empty:
                st.success("بر اساس معیارهای فعلی، موردی برای بررسی شناسایی نشد.")
            else:
                st.warning(
                    f"{len(attention_df)} دانش‌آموز بر اساس معیارهای اولیه "
                    "نیازمند بررسی انسانی هستند."
                )
                st.dataframe(
                    styled_table(attention_df),
                    use_container_width=True,
                    hide_index=True
                )

            # -------------------- RANKING / DATA EXPORT --------------------
            st.divider()
            section_title("گزارش عملکرد دانش‌آموزان")
            ranking_df = student_summary[[
                "student_id", "first_name", "last_name", "grade",
                "major", "class_name", "average_score",
                "attendance_percentage", "absent_days", "late_days",
                "academic_status"
            ]].copy()
            ranking_df = ranking_df.sort_values(
                "average_score", ascending=False
            )
            ranking_df.columns = [
                "شناسه", "نام", "نام خانوادگی", "پایه", "رشته",
                "کلاس", "میانگین نمرات", "درصد حضور",
                "تعداد غیبت", "تعداد تأخیر", "وضعیت بر اساس قواعد"
            ]
            ranking_df["میانگین نمرات"] = pd.to_numeric(ranking_df["میانگین نمرات"], errors="coerce").round(2)
            ranking_df["درصد حضور"] = pd.to_numeric(ranking_df["درصد حضور"], errors="coerce").round(1)
            st.dataframe(
                styled_table(ranking_df),
                use_container_width=True,
                hide_index=True
            )

            csv_bytes = ranking_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "⬇️ دریافت گزارش دانش‌آموزان (CSV)",
                data=csv_bytes,
                file_name="class_school_analytics.csv",
                mime="text/csv",
                use_container_width=True
            )

            # PDF report: charts, key indicators, selected filters and student table.
            subject_for_pdf = (
                filtered_grades.groupby("subject_name", as_index=False)["score"].mean()
                .rename(columns={"score": "average_score"})
                if not filtered_grades.empty else pd.DataFrame(columns=["subject_name", "average_score"])
            )
            status_names = {"present": "حاضر", "absent": "غایب", "late": "تأخیر"}
            status_for_pdf = (
                filtered_attendance["status"].map(status_names).fillna("نامشخص")
                .value_counts().rename_axis("status_label").reset_index(name="count")
                if not filtered_attendance.empty else pd.DataFrame(columns=["status_label", "count"])
            )
            pdf_metrics = {
                "students": total_students, "average": average_score,
                "attendance": attendance_rate, "absent": absent_count
            }
            pdf_filters = {"پایه": selected_grade, "رشته": selected_major, "کلاس": selected_class}
            try:
                pdf_bytes = build_class_analytics_pdf(
                    student_summary.sort_values("average_score", ascending=False),
                    subject_for_pdf, status_for_pdf, pdf_metrics, pdf_filters
                )
                st.download_button(
                    "📄 دریافت گزارش تحلیلی مدرسه (PDF)",
                    data=pdf_bytes,
                    file_name="class_school_analytics_report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            except ImportError as pdf_error:
                st.warning(
                    "برای ساخت PDF فارسی، کتابخانه‌های لازم را در ترمینال محیط پروژه نصب کنید: "
                    r".\venv\Scripts\python.exe -m pip install reportlab matplotlib arabic-reshaper python-bidi"
                )
            except Exception as pdf_error:
                st.error(f"ساخت گزارش PDF با خطا مواجه شد: {pdf_error}")

            st.caption(
                "محدودیت تحلیل: داده‌ها از سوابق ثبت‌شده فعلی استخراج می‌شوند. "
                "مقایسه نمرات، وضعیت‌های ساده مبتنی بر آستانه است و جایگزین "
                "ارزیابی حرفه‌ای هنرآموز یا مدیر نیست. اگر داده‌ها آزمایشی باشند، "
                "نتایج نیز صرفاً نمایشی و آموزشی‌اند."
            )


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
            if not user_can_access_student(current_user, student_id.strip()):
                st.error("شما به پرونده این دانش‌آموز دسترسی ندارید.")
                st.stop()

            # دریافت مسیر تصویر از پایگاه داده
            connection = get_connection()

            try:
                cursor = connection.cursor()

                cursor.execute("""
                    SELECT photo_path
                    FROM students
                    WHERE student_id = ?
                """, (student_id.strip(),))

                photo_record = cursor.fetchone()

            finally:
                connection.close()

            # نمایش تصویر دانش‌آموز
            if photo_record:
                photo_path = photo_record["photo_path"]
                display_student_photo(photo_path)

            # نمایش سایر اطلاعات پرونده
            show_student_report(student_id.strip())

        else:
            st.warning("لطفاً شناسه دانش‌آموز را وارد کنید.")



# ==========================================================
# ATTENDANCE MANAGEMENT
# ==========================================================
elif page == "مدیریت حضور و غیاب":

    hero(
        "مدیریت حضور و غیاب دانش‌آموزان",
        "ثبت وضعیت روزانه، مشاهده سوابق و اصلاح اطلاعات حضور و غیاب",
        "📅"
    )

    connection = get_connection()
    try:
        cursor = connection.cursor()
        students = get_accessible_students(current_user)
        cursor.execute("PRAGMA table_info(attendance)")
        attendance_columns = [row["name"] for row in cursor.fetchall()]
    finally:
        connection.close()

    date_column = next(
        (name for name in ("date", "attendance_date", "record_date")
         if name in attendance_columns),
        None
    )

    if not date_column:
        st.error("ستون تاریخ در جدول attendance پیدا نشد؛ ساختار جدول باید بررسی شود.")
    elif not students:
        st.warning("هنوز دانش‌آموزی در پایگاه داده ثبت نشده است.")
    else:
        student_options = {
            student["student_id"]: (
                f'{student["first_name"]} {student["last_name"]} '
                f'| {student["student_id"]} | {student["class_name"]}'
            )
            for student in students
        }

        selected_student_id = st.selectbox(
            "انتخاب دانش‌آموز:",
            options=list(student_options.keys()),
            format_func=lambda value: student_options[value],
            key="attendance_student_selector"
        )

        selected_student = next(
            item for item in students
            if item["student_id"] == selected_student_id
        )

        st.info(
            f'دانش‌آموز انتخاب‌شده: {selected_student["first_name"]} '
            f'{selected_student["last_name"]} | '
            f'پایه: {selected_student["grade"]} | '
            f'کلاس: {selected_student["class_name"]}'
        )

        st.divider()
        section_title("ثبت وضعیت روزانه")

        status_options = {
            "حاضر": "present",
            "غایب": "absent",
            "تأخیر": "late"
        }

        with st.form("attendance_entry_form"):
            attendance_date = st.date_input(
                "تاریخ حضور و غیاب:",
                key="attendance_date_input"
            )
            selected_status_label = st.radio(
                "وضعیت دانش‌آموز:",
                options=list(status_options.keys()),
                horizontal=True
            )
            save_attendance = st.form_submit_button(
                "💾 ثبت حضور و غیاب",
                use_container_width=True
            )

        if save_attendance:
            if not user_can_access_student(current_user, selected_student_id):
                st.error("اجازه ثبت حضور و غیاب این دانش‌آموز را ندارید.")
                st.stop()
            date_value = attendance_date.isoformat()
            connection = get_connection()
            try:
                cursor = connection.cursor()
                cursor.execute(
                    f'SELECT rowid FROM attendance WHERE student_id = ? AND "{date_column}" = ?',
                    (selected_student_id, date_value)
                )
                existing_record = cursor.fetchone()

                if existing_record:
                    st.warning(
                        "برای این دانش‌آموز در این تاریخ قبلاً وضعیت ثبت شده است. "
                        "برای تغییر آن از قسمت ویرایش سوابق استفاده کنید."
                    )
                else:
                    cursor.execute(
                        f'INSERT INTO attendance (student_id, "{date_column}", status) VALUES (?, ?, ?)',
                        (
                            selected_student_id,
                            date_value,
                            status_options[selected_status_label]
                        )
                    )
                    connection.commit()
                    st.success("اطلاعات حضور و غیاب با موفقیت ثبت شد.")
                    st.rerun()
            except Exception as exc:
                connection.rollback()
                st.error(f"خطا در ثبت حضور و غیاب: {exc}")
            finally:
                connection.close()

        st.divider()
        section_title("سوابق حضور و غیاب دانش‌آموز")

        connection = get_connection()
        try:
            query = (
                'SELECT rowid AS record_id, "' + date_column + '" AS attendance_date, status '
                'FROM attendance WHERE student_id = ? '
                'ORDER BY "' + date_column + '" DESC, rowid DESC'
            )
            records = pd.read_sql_query(
                query, connection, params=(selected_student_id,)
            )
        finally:
            connection.close()

        if records.empty:
            st.info("برای این دانش‌آموز هنوز سابقه حضور و غیابی ثبت نشده است.")
        else:
            status_labels = {
                "present": "حاضر",
                "absent": "غایب",
                "late": "تأخیر"
            }

            display_records = records.copy()
            display_records["status"] = (
                display_records["status"].map(status_labels).fillna("نامشخص")
            )
            display_records.columns = ["شناسه رکورد", "تاریخ", "وضعیت"]

            st.dataframe(
                styled_table(display_records.drop(columns=["شناسه رکورد"])),
                use_container_width=True,
                hide_index=True
            )

            present_count = int((records["status"] == "present").sum())
            absent_count = int((records["status"] == "absent").sum())
            late_count = int((records["status"] == "late").sum())

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("کل روزهای ثبت‌شده", len(records))
            m2.metric("حضور", present_count)
            m3.metric("غیبت", absent_count)
            m4.metric("تأخیر", late_count)

            st.divider()
            section_title("ویرایش یا حذف سابقه")

            record_options = {
                int(row["record_id"]): (
                    f'{row["attendance_date"]} | '
                    f'{status_labels.get(row["status"], "نامشخص")}'
                )
                for _, row in records.iterrows()
            }

            selected_record_id = st.selectbox(
                "سابقه موردنظر:",
                options=list(record_options.keys()),
                format_func=lambda value: record_options[value],
                key="attendance_record_selector"
            )

            selected_record = records[
                records["record_id"] == selected_record_id
            ].iloc[0]

            status_values = list(status_options.values())
            current_status_index = (
                status_values.index(selected_record["status"])
                if selected_record["status"] in status_values else 0
            )

            with st.form("edit_attendance_form"):
                new_status_label = st.selectbox(
                    "وضعیت جدید:",
                    options=list(status_options.keys()),
                    index=current_status_index
                )
                update_attendance = st.form_submit_button(
                    "✏️ ذخیره تغییرات",
                    use_container_width=True
                )

            if update_attendance:
                if not user_can_access_student(current_user, selected_student_id):
                    st.error("اجازه ویرایش حضور و غیاب این دانش‌آموز را ندارید.")
                    st.stop()
                connection = get_connection()
                try:
                    cursor = connection.cursor()
                    cursor.execute(
                        "UPDATE attendance SET status = ? WHERE rowid = ? AND student_id = ?",
                        (
                            status_options[new_status_label],
                            selected_record_id,
                            selected_student_id
                        )
                    )
                    connection.commit()
                    st.success("وضعیت حضور و غیاب ویرایش شد.")
                    st.rerun()
                except Exception as exc:
                    connection.rollback()
                    st.error(f"خطا در ویرایش سابقه: {exc}")
                finally:
                    connection.close()

            confirm_delete = st.checkbox(
                "حذف سابقه انتخاب‌شده را تأیید می‌کنم.",
                key=f"confirm_attendance_delete_{selected_record_id}"
            )

            if st.button(
                "🗑️ حذف سابقه انتخاب‌شده",
                disabled=not confirm_delete,
                use_container_width=True,
                key=f"delete_attendance_{selected_record_id}"
            ):
                connection = get_connection()
                try:
                    cursor = connection.cursor()
                    if not user_can_access_student(current_user, selected_student_id):
                        st.error("اجازه حذف سابقه حضور و غیاب را ندارید.")
                        st.stop()
                    cursor.execute(
                        "DELETE FROM attendance WHERE rowid = ? AND student_id = ?",
                        (selected_record_id, selected_student_id)
                    )
                    connection.commit()
                    st.success("سابقه حضور و غیاب حذف شد.")
                    st.rerun()
                except Exception as exc:
                    connection.rollback()
                    st.error(f"خطا در حذف سابقه: {exc}")
                finally:
                    connection.close()


# ==========================================================
# GRADE MANAGEMENT
# ==========================================================
elif page == "مدیریت نمرات":

    hero(
        "مدیریت نمرات دانش‌آموزان",
        "ثبت، مشاهده، ویرایش و حذف نمرات تحصیلی",
        "📝"
    )

    # دریافت فهرست درس‌ها
    connection = get_connection()

    try:
        cursor = connection.cursor()

        subjects = get_accessible_subjects(current_user)

    finally:
        connection.close()

    # مدیریت فهرست درس‌ها فقط برای مدیر و معاون
    if current_user["role"] != "teacher":
        with st.expander("➕ افزودن درس جدید"):

            with st.form("add_subject_form"):
                new_subject = st.text_input(
                    "نام درس جدید",
                    placeholder="مثلاً: برنامه‌نویسی"
                )

                add_subject_clicked = st.form_submit_button(
                    "ثبت درس",
                    use_container_width=True
                )

            if add_subject_clicked:

                if new_subject.strip():

                    connection = get_connection()

                    try:
                        cursor = connection.cursor()

                        cursor.execute("""
                            INSERT INTO subjects (subject_name)
                            VALUES (?)
                        """, (new_subject.strip(),))

                        connection.commit()

                        st.success("درس جدید با موفقیت اضافه شد.")
                        st.rerun()

                    except sqlite3.IntegrityError:
                        st.warning("این درس قبلاً ثبت شده است.")

                    finally:
                        connection.close()

                else:
                    st.warning("نام درس را وارد کنید.")

    st.divider()

    # دریافت فهرست دانش‌آموزان
    connection = get_connection()

    try:
        cursor = connection.cursor()

        students = get_accessible_students(current_user)

    finally:
        connection.close()

    if not students:

        st.warning("هنوز دانش‌آموزی در پایگاه داده ثبت نشده است.")

    else:

        st.subheader("👤 انتخاب دانش‌آموز")

        student_options = {
            student["student_id"]: (
                f'{student["first_name"]} {student["last_name"]} '
                f'| {student["student_id"]}'
            )
            for student in students
        }

        selected_student_id = st.selectbox(
            "دانش‌آموز موردنظر را انتخاب کنید:",
            options=list(student_options.keys()),
            format_func=lambda x: student_options[x]
        )

        selected_student = next(
            s for s in students
            if s["student_id"] == selected_student_id
        )

        st.info(
            f'نام: {selected_student["first_name"]} '
            f'{selected_student["last_name"]} | '
            f'پایه: {selected_student["grade"]} | '
            f'کلاس: {selected_student["class_name"]}'
        )

        # دریافت مجدد فهرست درس‌ها
        connection = get_connection()

        try:
            cursor = connection.cursor()

            subjects = get_accessible_subjects(current_user)

        finally:
            connection.close()

        # ثبت نمره جدید
        st.divider()
        st.subheader("➕ ثبت نمره جدید")

        if not subjects:

            st.warning(
                "ابتدا از قسمت افزودن درس جدید، حداقل یک درس ثبت کنید."
            )

        else:

            subject_options = {
                subject["subject_id"]: subject["subject_name"]
                for subject in subjects
            }

            with st.form("add_grade_form"):

                selected_subject = st.selectbox(
                    "نام درس:",
                    options=list(subject_options.keys()),
                    format_func=lambda x: subject_options[x]
                )

                term = st.selectbox(
                    "نوبت تحصیلی:",
                    [
                        "مستمر اول",
                        "پایانی اول",
                        "مستمر دوم",
                        "پایانی دوم"
                    ]
                )

                assessment = st.selectbox(
                    "نوع ارزشیابی:",
                    [
                        "آزمون کتبی",
                        "آزمون شفاهی",
                        "آزمون عملی",
                        "فعالیت کلاسی",
                        "پروژه"
                    ]
                )

                score = st.number_input(
                    "نمره:",
                    min_value=0.0,
                    max_value=20.0,
                    value=15.0,
                    step=0.25
                )

                exam_date = st.text_input(
                    "تاریخ ارزشیابی:",
                    placeholder="مثلاً 1405/08/20"
                )

                save_grade = st.form_submit_button(
                    "💾 ثبت نمره",
                    use_container_width=True
                )

            if save_grade:
                if not user_can_access_student(current_user, selected_student_id, selected_subject):
                    st.error("اجازه ثبت نمره برای این کلاس یا درس را ندارید.")
                    st.stop()

                connection = get_connection()

                try:
                    cursor = connection.cursor()

                    cursor.execute("""
                        INSERT INTO grades (
                            student_id,
                            subject_id,
                            term,
                            assessment,
                            score,
                            exam_date
                        )
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        selected_student_id,
                        selected_subject,
                        term,
                        assessment,
                        score,
                        exam_date.strip() or None
                    ))

                    connection.commit()

                    st.success("نمره با موفقیت ثبت شد.")
                    st.rerun()

                finally:
                    connection.close()

        # دریافت سوابق نمرات دانش‌آموز
        st.divider()
        st.subheader("📊 سوابق نمرات دانش‌آموز")

        connection = get_connection()

        try:
            cursor = connection.cursor()

            cursor.execute("""
                SELECT
                    grades.grade_id,
                    subjects.subject_name,
                    grades.term,
                    grades.assessment,
                    grades.score,
                    grades.exam_date
                FROM grades
                JOIN subjects
                    ON grades.subject_id = subjects.subject_id
                WHERE grades.student_id = ?
                  AND (
                      ? != 'teacher'
                      OR EXISTS (
                          SELECT 1 FROM teacher_assignments ta
                          JOIN students st ON st.student_id = grades.student_id
                          WHERE ta.user_id = ?
                            AND ta.class_name = st.class_name
                            AND ta.subject_id = grades.subject_id
                      )
                  )
                ORDER BY grades.grade_id DESC
            """, (
                selected_student_id, current_user["role"], current_user["user_id"]
            ))

            grade_records = [
                dict(row) for row in cursor.fetchall()
            ]

        finally:
            connection.close()

        if grade_records:

            display_records = [
                {
                    "شناسه نمره": row["grade_id"],
                    "درس": row["subject_name"],
                    "نوبت": row["term"],
                    "نوع ارزشیابی": row["assessment"],
                    "نمره": row["score"],
                    "تاریخ": row["exam_date"] or "-"
                }
                for row in grade_records
            ]

            st.dataframe(
                pd.DataFrame(display_records),
                use_container_width=True,
                hide_index=True
            )

            average_score = sum(
                row["score"] for row in grade_records
            ) / len(grade_records)

            st.metric(
                "میانگین نمرات ثبت‌شده",
                f"{average_score:.2f} از ۲۰"
            )

            # ویرایش نمره
            st.divider()
            st.subheader("✏️ ویرایش نمره")

            grade_options = {
                row["grade_id"]: (
                    f'{row["subject_name"]} | '
                    f'{row["term"]} | '
                    f'نمره: {row["score"]}'
                )
                for row in grade_records
            }

            selected_grade_id = st.selectbox(
                "نمره موردنظر برای ویرایش:",
                options=list(grade_options.keys()),
                format_func=lambda x: grade_options[x]
            )

            selected_grade = next(
                row for row in grade_records
                if row["grade_id"] == selected_grade_id
            )

            with st.form("edit_grade_form"):

                edit_subject = st.selectbox(
                    "درس:",
                    options=list(subject_options.keys()),
                    index=list(subject_options.keys()).index(
                        next(
                            s["subject_id"]
                            for s in subjects
                            if s["subject_name"] == selected_grade["subject_name"]
                        )
                    ),
                    format_func=lambda x: subject_options[x]
                )

                edit_term = st.selectbox(
                    "نوبت تحصیلی:",
                    [
                        "مستمر اول",
                        "پایانی اول",
                        "مستمر دوم",
                        "پایانی دوم"
                    ],
                    index=[
                        "مستمر اول",
                        "پایانی اول",
                        "مستمر دوم",
                        "پایانی دوم"
                    ].index(selected_grade["term"])
                )

                edit_assessment = st.selectbox(
                    "نوع ارزشیابی:",
                    [
                        "آزمون کتبی",
                        "آزمون شفاهی",
                        "آزمون عملی",
                        "فعالیت کلاسی",
                        "پروژه"
                    ],
                    index=[
                        "آزمون کتبی",
                        "آزمون شفاهی",
                        "آزمون عملی",
                        "فعالیت کلاسی",
                        "پروژه"
                    ].index(selected_grade["assessment"])
                )

                edit_score = st.number_input(
                    "نمره جدید:",
                    min_value=0.0,
                    max_value=20.0,
                    value=float(selected_grade["score"]),
                    step=0.25,
                    key=f"edit_score_{selected_grade_id}"
                )

                edit_date = st.text_input(
                    "تاریخ ارزشیابی:",
                    value=selected_grade["exam_date"] or "",
                    key=f"edit_date_{selected_grade_id}"
                )

                update_grade = st.form_submit_button(
                    "💾 ذخیره تغییرات",
                    use_container_width=True
                )

            if update_grade:
                if not user_can_access_student(current_user, selected_student_id, edit_subject):
                    st.error("اجازه ویرایش نمره در این کلاس یا درس را ندارید.")
                    st.stop()

                connection = get_connection()

                try:
                    cursor = connection.cursor()

                    cursor.execute("""
                        UPDATE grades
                        SET
                            subject_id = ?,
                            term = ?,
                            assessment = ?,
                            score = ?,
                            exam_date = ?
                        WHERE grade_id = ?
                        AND student_id = ?
                    """, (
                        edit_subject,
                        edit_term,
                        edit_assessment,
                        edit_score,
                        edit_date.strip() or None,
                        selected_grade_id,
                        selected_student_id
                    ))

                    connection.commit()

                    st.success("نمره با موفقیت ویرایش شد.")
                    st.rerun()

                finally:
                    connection.close()

            # حذف نمره
            st.divider()
            st.subheader("🗑️ حذف نمره")

            confirm_delete = st.checkbox(
                "حذف این نمره را تأیید می‌کنم.",
                key=f"confirm_delete_{selected_grade_id}"
            )

            if st.button(
                "حذف نمره انتخاب‌شده",
                type="primary",
                disabled=not confirm_delete,
                use_container_width=True
            ):

                selected_subject_id_for_delete = next(
                    s["subject_id"] for s in subjects
                    if s["subject_name"] == selected_grade["subject_name"]
                )
                if not user_can_access_student(
                    current_user, selected_student_id, selected_subject_id_for_delete
                ):
                    st.error("اجازه حذف این نمره را ندارید.")
                    st.stop()

                connection = get_connection()

                try:
                    cursor = connection.cursor()

                    cursor.execute("""
                        DELETE FROM grades
                        WHERE grade_id = ?
                        AND student_id = ?
                    """, (
                        selected_grade_id,
                        selected_student_id
                    ))

                    connection.commit()

                    st.success("نمره حذف شد.")
                    st.rerun()

                finally:
                    connection.close()

        else:

            st.info("برای این دانش‌آموز هنوز نمره‌ای ثبت نشده است.")

# ==========================================================
# STUDENT REGISTRATION
# ==========================================================
elif page == "ثبت دانش‌آموز":
    hero(
        "ثبت دانش‌آموز جدید",
        "ثبت مشخصات، تصاویر چهره و ایجاد QR اختصاصی",
        "📝"
    )

    section_title("اطلاعات دانش‌آموز")

    # اگر کاربر از بخش تشخیص چهره به این صفحه آمده باشد،
    # تصویر چهره ناشناس در این متغیر موجود است.
    unknown_photo_bytes = st.session_state.get("unknown_face_photo")

    if unknown_photo_bytes:
        st.info(
            "تصویر چهره ناشناس به‌عنوان تصویر اول انتخاب شده است. "
            "برای تکمیل ثبت‌نام، دو تصویر دیگر از همان دانش‌آموز بارگذاری کنید."
        )
        st.image(
            unknown_photo_bytes,
            caption="تصویر منتقل‌شده از بخش تشخیص چهره",
            width=220
        )

    with st.form("student_registration_form"):

        col1, col2 = st.columns(2)

        with col1:
            student_id = st.text_input(
                "شناسه دانش‌آموز",
                placeholder="مثلاً 11001"
            )
            first_name = st.text_input("نام")
            grade = st.selectbox(
                "پایه تحصیلی",
                ["دهم", "یازدهم", "دوازدهم"]
            )

        with col2:
            last_name = st.text_input("نام خانوادگی")
            major = st.text_input(
                "رشته تحصیلی",
                value="شبکه و نرم‌افزار رایانه"
            )
            class_name = st.text_input(
                "نام کلاس",
                placeholder="مثلاً ۱۰ کامپیوتر - الف"
            )

        st.markdown("---")
        st.subheader("📸 تصاویر چهره دانش‌آموز")

        required_uploads = 2 if unknown_photo_bytes else 3
        uploader_label = (
            "دو تصویر واضح دیگر از چهره دانش‌آموز انتخاب کنید"
            if unknown_photo_bytes
            else "حداقل ۳ تصویر واضح از چهره دانش‌آموز انتخاب کنید"
        )

        uploaded_photos = st.file_uploader(
            uploader_label,
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            help="ترجیحاً تصاویر با زاویه و نور متفاوت انتخاب شوند."
        )

        uploaded_count = len(uploaded_photos) if uploaded_photos else 0
        total_photo_count = uploaded_count + (1 if unknown_photo_bytes else 0)

        st.info(
            f"تعداد کل تصاویر: {total_photo_count} از ۳"
        )

        st.markdown("---")

        submitted = st.form_submit_button(
            "ثبت اطلاعات دانش‌آموز",
            use_container_width=True
        )

    if submitted:
        if not all([
            student_id.strip(),
            first_name.strip(),
            last_name.strip(),
            major.strip(),
            class_name.strip()
        ]):
            st.warning("لطفاً تمام اطلاعات را تکمیل کنید.")

        elif uploaded_count < required_uploads:
            st.warning(
                f"لطفاً حداقل {required_uploads} تصویر دیگر بارگذاری کنید."
            )

        else:
            images = []
            image_error = False

            # ابتدا تصویر منتقل‌شده از بخش تشخیص چهره به فهرست اضافه می‌شود.
            if unknown_photo_bytes:
                first_image = cv2.imdecode(
                    np.frombuffer(unknown_photo_bytes, dtype=np.uint8),
                    cv2.IMREAD_COLOR
                )

                if first_image is None:
                    st.error("تصویر منتقل‌شده از بخش تشخیص چهره قابل خواندن نیست.")
                    image_error = True
                else:
                    images.append(first_image)

            # سپس تصاویر جدید بارگذاری‌شده اضافه می‌شوند.
            if not image_error:
                for photo in uploaded_photos:
                    image = cv2.imdecode(
                        np.frombuffer(photo.getvalue(), dtype=np.uint8),
                        cv2.IMREAD_COLOR
                    )

                    if image is None:
                        st.error(f"تصویر {photo.name} قابل خواندن نیست.")
                        image_error = True
                        break

                    images.append(image)

            if not image_error and len(images) >= 3:
                try:
                    result = register_student(
                        student_id=student_id.strip(),
                        first_name=first_name.strip(),
                        last_name=last_name.strip(),
                        grade=grade,
                        major=major.strip(),
                        class_name=class_name.strip(),
                        images=images
                    )

                    st.success("دانش‌آموز با موفقیت ثبت شد!")

                    st.write(
                        "شناسه دانش‌آموز:",
                        result["student_id"]
                    )
                    st.write(
                        "تعداد تصاویر:",
                        result["number_of_photos"]
                    )
                    st.write(
                        "تعداد الگوهای چهره:",
                        result["number_of_templates"]
                    )

                    st.image(
                        result["qr_path"],
                        caption="QR اختصاصی دانش‌آموز",
                        width=220
                    )

                    # جلوگیری از استفاده مجدد ناخواسته از تصویر در ثبت بعدی
                    st.session_state.pop("unknown_face_photo", None)

                except Exception as e:
                    st.error(f"خطا در ثبت دانش‌آموز: {e}")

# ==========================================================
# IMAGE PROCESSING & QR
# ==========================================================

# ==========================================================
# GRADE MANAGEMENT
# ==========================================================
elif page == "مدیریت نمرات":

    hero(
        "مدیریت نمرات دانش‌آموز",
        "ثبت، مشاهده، ویرایش و حذف نمرات تحصیلی",
        "📚"
    )

    # دریافت فهرست دانش‌آموزان
    connection = get_connection()

    try:
        cursor = connection.cursor()

        cursor.execute("""
            SELECT student_id, first_name, last_name, grade, class_name
            FROM students
            ORDER BY last_name, first_name
        """)

        students = cursor.fetchall()

        cursor.execute("""
            SELECT subject_id, subject_name
            FROM subjects
            ORDER BY subject_name
        """)

        subjects = cursor.fetchall()

    finally:
        connection.close()

    if not students:
        st.warning("هنوز دانش‌آموزی در پایگاه داده ثبت نشده است.")

    else:

        student_options = {
            f"{row['student_id']} | {row['first_name']} {row['last_name']}": row['student_id']
            for row in students
        }

        selected_student_label = st.selectbox(
            "انتخاب دانش‌آموز:",
            list(student_options.keys()),
            key="grade_student_selector"
        )

        selected_student_id = student_options[selected_student_label]

        selected_student = next(
            row for row in students
            if row["student_id"] == selected_student_id
        )

        st.success(
            f"دانش‌آموز انتخاب‌شده: "
            f"{selected_student['first_name']} "
            f"{selected_student['last_name']} | "
            f"پایه: {selected_student['grade']} | "
            f"کلاس: {selected_student['class_name']}"
        )

        st.divider()

        # --------------------------------------------------
        # ADD NEW SUBJECT
        # --------------------------------------------------
        with st.expander("➕ افزودن درس جدید"):

            with st.form("add_subject_form"):

                new_subject = st.text_input(
                    "نام درس جدید:"
                )

                add_subject_clicked = st.form_submit_button(
                    "ثبت درس",
                    use_container_width=True
                )

            if add_subject_clicked:

                if not new_subject.strip():
                    st.warning("نام درس را وارد کنید.")

                else:
                    connection = get_connection()

                    try:
                        cursor = connection.cursor()

                        cursor.execute("""
                            INSERT INTO subjects (subject_name)
                            VALUES (?)
                        """, (new_subject.strip(),))

                        connection.commit()

                        st.success("درس جدید با موفقیت ثبت شد.")
                        st.rerun()

                    except sqlite3.IntegrityError:
                        st.warning("این درس قبلاً ثبت شده است.")

                    finally:
                        connection.close()

        # --------------------------------------------------
        # ADD GRADE
        # --------------------------------------------------
        section_title("ثبت نمره جدید")

        if not subjects:
            st.warning(
                "هنوز هیچ درسی در پایگاه داده ثبت نشده است. "
                "ابتدا یک درس جدید اضافه کنید."
            )

        else:

            subject_options = {
                row["subject_name"]: row["subject_id"]
                for row in subjects
            }

            with st.form("add_grade_form"):

                selected_subject = st.selectbox(
                    "نام درس:",
                    list(subject_options.keys())
                )

                col1, col2 = st.columns(2)

                with col1:
                    selected_term = st.selectbox(
                        "نوبت تحصیلی:",
                        [
                            "مستمر اول",
                            "پایانی اول",
                            "مستمر دوم",
                            "پایانی دوم"
                        ]
                    )

                with col2:
                    selected_assessment = st.selectbox(
                        "نوع ارزشیابی:",
                        [
                            "آزمون کتبی",
                            "آزمون شفاهی",
                            "آزمون عملی",
                            "فعالیت کلاسی",
                            "پروژه"
                        ]
                    )

                col3, col4 = st.columns(2)

                with col3:
                    score = st.number_input(
                        "نمره:",
                        min_value=0.0,
                        max_value=20.0,
                        value=15.0,
                        step=0.25
                    )

                with col4:
                    exam_date = st.date_input(
                        "تاریخ ارزشیابی:"
                    )

                save_grade = st.form_submit_button(
                    "💾 ثبت نمره",
                    use_container_width=True
                )

            if save_grade:

                connection = get_connection()

                try:
                    cursor = connection.cursor()

                    cursor.execute("""
                        INSERT INTO grades (
                            student_id,
                            subject_id,
                            term,
                            assessment,
                            score,
                            exam_date
                        )
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        selected_student_id,
                        subject_options[selected_subject],
                        selected_term,
                        selected_assessment,
                        score,
                        exam_date.isoformat()
                    ))

                    connection.commit()

                    st.success("نمره با موفقیت ثبت شد.")
                    st.rerun()

                except Exception as e:
                    st.error(f"خطا در ثبت نمره: {e}")

                finally:
                    connection.close()

        st.divider()

        # --------------------------------------------------
        # DISPLAY GRADES
        # --------------------------------------------------
        section_title("سوابق نمرات دانش‌آموز")

        connection = get_connection()

        try:
            grades_df = pd.read_sql_query("""
                SELECT
                    grades.grade_id AS 'شناسه نمره',
                    subjects.subject_name AS 'نام درس',
                    grades.term AS 'نوبت تحصیلی',
                    grades.assessment AS 'نوع ارزشیابی',
                    grades.score AS 'نمره',
                    grades.exam_date AS 'تاریخ'
                FROM grades
                INNER JOIN subjects
                    ON grades.subject_id = subjects.subject_id
                WHERE grades.student_id = ?
                ORDER BY grades.exam_date DESC, grades.grade_id DESC
            """, connection, params=(selected_student_id,))

        finally:
            connection.close()

        if grades_df.empty:
            st.info("برای این دانش‌آموز هنوز نمره‌ای ثبت نشده است.")

        else:

            st.dataframe(
                grades_df.drop(columns=["شناسه نمره"]),
                use_container_width=True,
                hide_index=True
            )

            st.metric(
                "میانگین نمرات ثبت‌شده",
                f"{grades_df['نمره'].mean():.2f}"
            )

            st.divider()

            # --------------------------------------------------
            # EDIT GRADE
            # --------------------------------------------------
            section_title("ویرایش یا حذف نمره")

            grade_ids = grades_df["شناسه نمره"].tolist()

            selected_grade_id = st.selectbox(
                "نمره موردنظر را انتخاب کنید:",
                grade_ids,
                format_func=lambda grade_id: (
                    f"نمره شماره {grade_id} | "
                    f"{grades_df.loc[grades_df['شناسه نمره'] == grade_id, 'نام درس'].iloc[0]} | "
                    f"{grades_df.loc[grades_df['شناسه نمره'] == grade_id, 'نمره'].iloc[0]}"
                )
            )

            selected_grade = grades_df[
                grades_df["شناسه نمره"] == selected_grade_id
            ].iloc[0]

            with st.form("edit_grade_form"):

                edit_score = st.number_input(
                    "نمره جدید:",
                    min_value=0.0,
                    max_value=20.0,
                    value=float(selected_grade["نمره"]),
                    step=0.25
                )

                edit_term = st.selectbox(
                    "نوبت تحصیلی:",
                    [
                        "مستمر اول",
                        "پایانی اول",
                        "مستمر دوم",
                        "پایانی دوم"
                    ],
                    index=[
                        "مستمر اول",
                        "پایانی اول",
                        "مستمر دوم",
                        "پایانی دوم"
                    ].index(selected_grade["نوبت تحصیلی"])
                    if selected_grade["نوبت تحصیلی"] in [
                        "مستمر اول",
                        "پایانی اول",
                        "مستمر دوم",
                        "پایانی دوم"
                    ] else 0
                )

                edit_assessment = st.text_input(
                    "نوع ارزشیابی:",
                    value=str(selected_grade["نوع ارزشیابی"])
                )

                edit_grade_clicked = st.form_submit_button(
                    "✏️ ذخیره تغییرات",
                    use_container_width=True
                )

            if edit_grade_clicked:

                connection = get_connection()

                try:
                    cursor = connection.cursor()

                    cursor.execute("""
                        UPDATE grades
                        SET score = ?,
                            term = ?,
                            assessment = ?
                        WHERE grade_id = ?
                          AND student_id = ?
                    """, (
                        edit_score,
                        edit_term,
                        edit_assessment.strip(),
                        selected_grade_id,
                        selected_student_id
                    ))

                    connection.commit()

                    st.success("نمره با موفقیت ویرایش شد.")
                    st.rerun()

                finally:
                    connection.close()

            if st.button(
                "🗑️ حذف نمره انتخاب‌شده",
                use_container_width=True
            ):

                st.session_state["confirm_grade_delete"] = selected_grade_id

            if st.session_state.get("confirm_grade_delete") == selected_grade_id:

                st.warning("آیا از حذف این نمره مطمئن هستید؟")

                col_yes, col_no = st.columns(2)

                with col_yes:
                    if st.button("بله، حذف شود", key="confirm_delete_yes"):

                        connection = get_connection()

                        try:
                            cursor = connection.cursor()

                            cursor.execute("""
                                DELETE FROM grades
                                WHERE grade_id = ?
                                  AND student_id = ?
                            """, (
                                selected_grade_id,
                                selected_student_id
                            ))

                            connection.commit()

                        finally:
                            connection.close()

                        st.session_state.pop("confirm_grade_delete", None)
                        st.success("نمره حذف شد.")
                        st.rerun()

                with col_no:
                    if st.button("انصراف", key="confirm_delete_no"):
                        st.session_state.pop("confirm_grade_delete", None)
                        st.rerun()

elif page == "پردازش تصویر و QR":
    hero(
        "پردازش تصویر و شناسایی دانش‌آموز",
        "تشخیص چهره از تصویر و بازیابی پرونده با QR Code",
        "📷"
    )

    face_tab, qr_tab = st.tabs(["🙂 تشخیص چهره", "🔳 خواندن QR Code"])

    with face_tab:
        
        section_title("تشخیص و شناسایی هویت دانش‌آموز")

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
            uploaded_image = st.camera_input(
                "تصویر را با دوربین ثبت کنید:"
            )

        if uploaded_image is not None:

            pil_image = Image.open(uploaded_image).convert("RGB")
            image_array = np.array(pil_image)
            image_bgr = cv2.cvtColor(
                image_array,
                cv2.COLOR_RGB2BGR
            )

            # مرحله اول: تشخیص چهره
            result = detect_faces(image_bgr)

            st.image(
                cv2.cvtColor(
                    result["image"],
                    cv2.COLOR_BGR2RGB
                ),
                caption="نتیجه تشخیص چهره",
                use_container_width=True
            )

            st.metric(
                "تعداد چهره‌های تشخیص داده‌شده",
                result["face_count"]
            )

            if result["face_count"] == 0:
                st.warning("هیچ چهره‌ای در تصویر پیدا نشد.")

            elif result["face_count"] > 1:
                st.warning(
                    "بیش از یک چهره در تصویر وجود دارد. "
                    "لطفاً تصویر یک نفر را انتخاب کنید."
                )

            else:
                try:
                    # مرحله دوم: استخراج ویژگی چهره
                    recognizer = FaceRecognition()

                    query_feature = recognizer.extract_feature(
                        image_bgr
                    )

                    # مرحله سوم: دریافت الگوهای ذخیره‌شده
                    student_features = get_all_face_templates()

                    if not student_features:
                        st.warning(
                            "هیچ الگوی چهره‌ای در پایگاه داده ثبت نشده است."
                        )

                    else:
                        # مرحله چهارم: شناسایی دانش‌آموز
                        identification = recognizer.identify(
                            query_feature,
                            student_features
                        )

                        status = identification["status"]

                        if status == "IDENTIFIED":

                            student_id = identification["student_id"]

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
                                        class_name,
                                        photo_path
                                    FROM students
                                    WHERE student_id = ?
                                """, (student_id,))

                                student = cursor.fetchone()

                            finally:
                                connection.close()

                            if student:
                                student = dict(student)

                                st.success(
                                    "دانش‌آموز با موفقیت شناسایی شد."
                                )

                                st.write(
                                    f"امتیاز شباهت: "
                                    f"{identification['score']:.3f}"
                                )

                                col_photo, col_info = st.columns([1, 3])

                                with col_photo:
                                    photo_path = student.get("photo_path")

                                    if photo_path:
                                        full_photo_path = BASE_DIR / photo_path

                                        if full_photo_path.exists():
                                            st.image(
                                                str(full_photo_path),
                                                caption="تصویر ثبت‌شده",
                                                width=220
                                            )
                                        else:
                                            st.warning(
                                                "فایل تصویر ثبت‌شده پیدا نشد."
                                            )

                                with col_info:
                                    q1, q2, q3 = st.columns(3)

                                    q1.metric(
                                        "شناسه",
                                        student["student_id"]
                                    )

                                    q2.metric(
                                        "پایه",
                                        student["grade"]
                                    )

                                    q3.metric(
                                        "کلاس",
                                        student["class_name"]
                                    )

                                    st.write(
                                        f"**نام:** "
                                        f"{student['first_name']} "
                                        f"{student['last_name']}"
                                    )

                                    st.write(
                                        f"**رشته:** {student['major']}"
                                    )

                                st.divider()

                                show_student_report(
                                    student["student_id"]
                                )

                            else:
                                st.error(
                                    "الگوی چهره پیدا شد، اما پرونده دانش‌آموز "
                                    "در پایگاه داده موجود نیست."
                                )

                        
                        elif status == "MANUAL_REVIEW":

                            st.warning(
                                "نتیجه شناسایی قطعی نیست و به بررسی انسانی نیاز دارد."
                            )

                            st.write(
                                f"بیشترین امتیاز شباهت: "
                                f"{identification['score']:.3f}"
                            )

                            st.write(
                                f"اختلاف با گزینه بعدی: "
                                f"{identification['margin']:.3f}"
                            )

                        
                        else:

                            st.markdown(
                                """
                                <div class="unknown-face-message">
                                    ⚠️ این چهره با اطمینان کافی در پایگاه داده شناسایی نشد.
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                            st.write(
                                f"بیشترین امتیاز شباهت: "
                                f"{identification['score']:.3f}"
                            )

                            st.button(
                                "📝 ثبت این دانش‌آموز در سامانه",
                                key="register_unknown_student",
                                use_container_width=True,
                                on_click=go_to_registration,
                                args=(uploaded_image.getvalue(),)
                            )

                            


                            

                except Exception as e:
                    st.error(
                        f"خطا در پردازش یا شناسایی چهره: {e}"
                    )

            st.caption(
                "نتایج شناسایی آزمایشی هستند و برای تصمیم‌گیری قطعی "
                "باید توسط فرد مسئول بررسی شوند."
            )

   
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

                    col_photo, col_info = st.columns([1, 3])

                    # نمایش تصویر دانش‌آموز
                    with col_photo:
                        photo_path = student.get("photo_path")

                        if photo_path:
                            full_photo_path = BASE_DIR / photo_path

                            if full_photo_path.exists():
                                st.image(
                                    str(full_photo_path),
                                    caption="تصویر دانش‌آموز",
                                    width=220
                                )
                            else:
                                st.warning("فایل تصویر پیدا نشد.")
                        else:
                            st.info("تصویری برای این دانش‌آموز ثبت نشده است.")

                    # نمایش مشخصات دانش‌آموز
                    with col_info:
                        q1, q2, q3 = st.columns(3)

                        q1.metric("شناسه", student["student_id"])
                        q2.metric("پایه", student["grade"])
                        q3.metric("کلاس", student["class_name"])

                        st.write(
                            f"**نام:** {student['first_name']} {student['last_name']}"
                        )
                        st.write(f"**رشته:** {student['major']}")

                    st.divider()
                    show_student_report(student["student_id"])

                else:
                    st.error("برای این QR Code پرونده‌ای پیدا نشد.")

            else:
                st.error(
                    "QR Code قابل شناسایی نیست. لطفاً تصویر واضح‌تری انتخاب کنید."
                )


# ==========================================================
# USER MANAGEMENT (ADMIN ONLY)
# ==========================================================
if page == "مدیریت کاربران":
    if current_user["role"] != "admin":
        st.error("شما اجازه دسترسی به مدیریت کاربران را ندارید.")
        st.stop()

    hero(
        "مدیریت کاربران سامانه",
        "ایجاد حساب، تعیین نقش، فعال یا غیرفعال‌سازی و بازنشانی رمز عبور",
        "🛡️"
    )

    role_names = {
        "admin": "مدیر سامانه",
        "deputy": "معاون آموزشی",
        "teacher": "هنرآموز"
    }

    tab_create, tab_list, tab_security, tab_assign = st.tabs(
        ["ایجاد حساب جدید", "فهرست کاربران", "مدیریت حساب‌ها", "تخصیص کلاس و درس"]
    )

    with tab_create:
        st.subheader("ایجاد حساب کاربری")
        with st.form("create_managed_user", clear_on_submit=True):
            new_full_name = st.text_input("نام و نام خانوادگی")
            new_username = st.text_input("نام کاربری")
            new_role_label = st.selectbox(
                "نقش کاربر",
                ["معاون آموزشی", "هنرآموز", "مدیر سامانه"]
            )
            new_password = st.text_input("رمز عبور (حداقل ۱۰ نویسه)", type="password")
            new_password2 = st.text_input("تکرار رمز عبور", type="password")
            submit_new_user = st.form_submit_button("ایجاد حساب", use_container_width=True)

        if submit_new_user:
            role_map = {
                "معاون آموزشی": "deputy",
                "هنرآموز": "teacher",
                "مدیر سامانه": "admin"
            }
            if new_password != new_password2:
                st.error("رمز عبور و تکرار آن یکسان نیستند.")
            else:
                try:
                    create_user(
                        new_username, new_full_name, new_password,
                        role=role_map[new_role_label]
                    )
                    st.success("حساب کاربری با موفقیت ایجاد شد.")
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
                except Exception as exc:
                    if "UNIQUE" in str(exc).upper():
                        st.error("این نام کاربری قبلاً ثبت شده است.")
                    else:
                        st.error(f"ایجاد حساب انجام نشد: {exc}")

    with tab_list:
        st.subheader("حساب‌های ثبت‌شده")
        users = list_users()
        if users:
            users_df = pd.DataFrame(users)
            users_df["role"] = users_df["role"].map(role_names).fillna(users_df["role"])
            users_df["is_active"] = users_df["is_active"].map({1: "فعال", 0: "غیرفعال"})
            users_df["must_change_password"] = users_df["must_change_password"].map(
                {1: "در انتظار تغییر رمز", 0: "عادی"}
            )
            users_df = users_df.rename(columns={
                "user_id": "شناسه",
                "username": "نام کاربری",
                "full_name": "نام و نام خانوادگی",
                "role": "نقش",
                "is_active": "وضعیت",
                "created_at": "تاریخ ایجاد",
                "password_changed_at": "آخرین تغییر رمز",
                "must_change_password": "وضعیت رمز عبور"
            })
            st.dataframe(styled_table(users_df), use_container_width=True, hide_index=True)
        else:
            st.info("هنوز حسابی ثبت نشده است.")

    with tab_security:
        st.subheader("تغییر وضعیت، نقش یا رمز عبور")
        users = list_users()
        if not users:
            st.info("حسابی برای مدیریت وجود ندارد.")
        else:
            user_options = {
                f"{u['full_name']} — {u['username']} (شناسه {u['user_id']})": u
                for u in users
            }
            selected_label = st.selectbox("انتخاب حساب", list(user_options.keys()))
            selected_user = user_options[selected_label]
            st.caption(
                f"نقش فعلی: {role_names.get(selected_user['role'], selected_user['role'])} | "
                f"وضعیت: {'فعال' if selected_user['is_active'] else 'غیرفعال'} | "
                f"آخرین تغییر رمز: {selected_user.get('password_changed_at') or 'ثبت نشده'}"
            )

            col_role, col_active = st.columns(2)
            with col_role:
                selected_role_label = st.selectbox(
                    "نقش جدید",
                    ["مدیر سامانه", "معاون آموزشی", "هنرآموز"],
                    index={"admin": 0, "deputy": 1, "teacher": 2}.get(selected_user["role"], 0),
                    key=f"role_{selected_user['user_id']}"
                )
                if st.button("ذخیره نقش", key=f"save_role_{selected_user['user_id']}"):
                    role_value = {
                        "مدیر سامانه": "admin",
                        "معاون آموزشی": "deputy",
                        "هنرآموز": "teacher"
                    }[selected_role_label]
                    try:
                        change_user_role(selected_user["user_id"], role_value)
                        st.success("نقش کاربر به‌روزرسانی شد.")
                        st.rerun()
                    except ValueError as exc:
                        st.error(str(exc))

            with col_active:
                desired_active = not bool(selected_user["is_active"])
                action_label = "فعال‌سازی حساب" if desired_active else "غیرفعال‌سازی حساب"
                if st.button(action_label, key=f"toggle_active_{selected_user['user_id']}"):
                    try:
                        set_user_active(
                            selected_user["user_id"], desired_active,
                            current_user["user_id"]
                        )
                        st.success("وضعیت حساب تغییر کرد.")
                        st.rerun()
                    except ValueError as exc:
                        st.error(str(exc))

            st.divider()
            st.markdown("#### بازنشانی رمز عبور")
            st.caption("رمز فعلی قابل مشاهده نیست. می‌توانید رمز دلخواه تعیین کنید یا رمز موقت امن بسازید.")

            with st.form(f"reset_password_{selected_user['user_id']}"):
                reset_password = st.text_input("رمز جدید دلخواه (حداقل ۱۰ نویسه)", type="password")
                reset_password_confirm = st.text_input("تکرار رمز جدید", type="password")
                reset_submit = st.form_submit_button("ثبت رمز جدید")
            if reset_submit:
                if reset_password != reset_password_confirm:
                    st.error("رمز عبور جدید و تکرار آن یکسان نیستند.")
                else:
                    try:
                        reset_user_password(
                            selected_user["user_id"], reset_password, force_change=True
                        )
                        st.success("رمز جدید ثبت شد؛ کاربر در ورود بعدی باید آن را تغییر دهد.")
                        st.rerun()
                    except ValueError as exc:
                        st.error(str(exc))

            temp_key = f"temporary_password_{selected_user['user_id']}"
            if st.button("🎲 تولید رمز موقت تصادفی", key=f"generate_temp_{selected_user['user_id']}"):
                try:
                    temp_password = generate_temporary_password(selected_user["user_id"])
                    st.session_state[temp_key] = temp_password
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))

            if st.session_state.get(temp_key):
                st.success("رمز موقت تولید شد. آن را فقط همین حالا به کاربر تحویل دهید؛ پس از خروج یا تازه‌سازی صفحه دوباره نمایش داده نمی‌شود.")
                st.code(st.session_state[temp_key], language=None)
                if st.button("پنهان کردن رمز موقت", key=f"hide_temp_{selected_user['user_id']}"):
                    st.session_state.pop(temp_key, None)
                    st.rerun()


    with tab_assign:
        st.subheader("تخصیص کلاس‌ها و درس‌ها به هنرآموز")
        teachers = [
            u for u in list_users()
            if u["role"] == "teacher" and u["is_active"]
        ]
        if not teachers:
            st.info("ابتدا یک حساب فعال با نقش هنرآموز ایجاد کنید.")
        else:
            teacher_map = {
                f'{u["full_name"]} — {u["username"]}': u for u in teachers
            }
            teacher_label = st.selectbox(
                "انتخاب هنرآموز", list(teacher_map.keys())
            )
            teacher = teacher_map[teacher_label]
            connection = get_connection()
            try:
                class_rows = connection.execute("""
                    SELECT DISTINCT class_name FROM students
                    WHERE class_name IS NOT NULL AND TRIM(class_name) != ''
                    ORDER BY class_name
                """).fetchall()
                subject_rows = connection.execute("""
                    SELECT subject_id, subject_name FROM subjects ORDER BY subject_name
                """).fetchall()
            finally:
                connection.close()

            class_names = [r["class_name"] for r in class_rows]
            subject_map = {
                int(r["subject_id"]): r["subject_name"] for r in subject_rows
            }
            existing = get_teacher_assignments(teacher["user_id"])
            existing_pairs = {
                (r["class_name"], int(r["subject_id"])) for r in existing
            }

            if not class_names or not subject_map:
                st.warning("برای تخصیص، ابتدا کلاس و درس در سامانه ثبت شده باشد.")
            else:
                st.caption(
                    "برای هر هنرآموز، ترکیب دقیق کلاس و درس را تعیین کنید. "
                    "هنرآموز فقط به همین ترکیب‌ها دسترسی خواهد داشت."
                )
                chosen_pairs = []
                for ci, class_name in enumerate(class_names):
                    with st.expander(
                        f"کلاس {class_name}",
                        expanded=any(c == class_name for c, _ in existing_pairs)
                    ):
                        chosen_ids = st.multiselect(
                            f"درس‌های مجاز در کلاس {class_name}",
                            options=list(subject_map.keys()),
                            default=[
                                sid for c, sid in existing_pairs if c == class_name
                            ],
                            format_func=lambda sid: subject_map[sid],
                            key=f"teacher_assignment_{teacher['user_id']}_{ci}"
                        )
                        chosen_pairs.extend(
                            (class_name, sid) for sid in chosen_ids
                        )

                if st.button(
                    "ذخیره تخصیص‌ها", type="primary",
                    use_container_width=True, key="save_teacher_assignments"
                ):
                    try:
                        set_teacher_assignments(teacher["user_id"], chosen_pairs)
                        st.success("تخصیص کلاس‌ها و درس‌ها ذخیره شد.")
                        st.rerun()
                    except ValueError as exc:
                        st.error(str(exc))

                if existing:
                    st.markdown("**تخصیص‌های فعلی:**")
                    st.dataframe(
                        pd.DataFrame([
                            {"کلاس": r["class_name"], "درس": r["subject_name"]}
                            for r in existing
                        ]),
                        use_container_width=True, hide_index=True
                    )


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
