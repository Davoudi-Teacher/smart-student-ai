"""Authentication helpers for Smart Student AI.

Passwords are stored using PBKDF2-HMAC-SHA256 with a unique random salt.
No plain-text password is written to SQLite.
"""

import string
import hashlib
import hmac
import secrets
from modules.database_manager import get_connection

_ITERATIONS = 310_000


def initialize_auth_table():
    connection = get_connection()
    try:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS app_users (
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                full_name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                password_salt TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'admin'
                    CHECK(role IN ('admin', 'deputy', 'teacher')),
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                must_change_password INTEGER NOT NULL DEFAULT 0,
                password_changed_at TEXT
            )
        """)
        # Backward-compatible migration for accounts created by earlier versions.
        columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(app_users)").fetchall()
        }
        if "must_change_password" not in columns:
            connection.execute(
                "ALTER TABLE app_users ADD COLUMN must_change_password INTEGER NOT NULL DEFAULT 0"
            )
        if "password_changed_at" not in columns:
            connection.execute("ALTER TABLE app_users ADD COLUMN password_changed_at TEXT")
            connection.execute(
                "UPDATE app_users SET password_changed_at = created_at WHERE password_changed_at IS NULL"
            )
        connection.commit()
    finally:
        connection.close()


def count_users():
    connection = get_connection()
    try:
        return connection.execute("SELECT COUNT(*) FROM app_users").fetchone()[0]
    finally:
        connection.close()


def _hash_password(password: str, salt_hex: str) -> str:
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), _ITERATIONS
    )
    return digest.hex()


def create_user(username: str, full_name: str, password: str, role: str = "admin"):
    username = username.strip()
    full_name = full_name.strip()
    if len(username) < 4:
        raise ValueError("نام کاربری باید حداقل ۴ نویسه داشته باشد.")
    if len(full_name) < 2:
        raise ValueError("نام و نام خانوادگی را کامل وارد کنید.")
    if len(password) < 10:
        raise ValueError("رمز عبور باید حداقل ۱۰ نویسه داشته باشد.")
    if role not in {"admin", "deputy", "teacher"}:
        raise ValueError("نقش کاربر معتبر نیست.")

    salt = secrets.token_hex(16)
    password_hash = _hash_password(password, salt)
    connection = get_connection()
    try:
        connection.execute(
            "INSERT INTO app_users (username, full_name, password_hash, password_salt, role) VALUES (?, ?, ?, ?, ?)",
            (username, full_name, password_hash, salt, role),
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def authenticate_user(username: str, password: str):
    connection = get_connection()
    try:
        row = connection.execute(
            "SELECT user_id, username, full_name, password_hash, password_salt, role, is_active, "
            "must_change_password FROM app_users WHERE username = ? COLLATE NOCASE",
            (username.strip(),),
        ).fetchone()
    finally:
        connection.close()

    if row is None or not row["is_active"]:
        return None
    candidate = _hash_password(password, row["password_salt"])
    if not hmac.compare_digest(candidate, row["password_hash"]):
        return None
    return {
        "user_id": row["user_id"],
        "username": row["username"],
        "full_name": row["full_name"],
        "role": row["role"],
        "must_change_password": bool(row["must_change_password"]),
    }


def list_users():
    """Return account metadata only; never expose password hashes or salts."""
    connection = get_connection()
    try:
        rows = connection.execute(
            "SELECT user_id, username, full_name, role, is_active, created_at, "
            "password_changed_at, must_change_password "
            "FROM app_users ORDER BY user_id"
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        connection.close()


def set_user_active(user_id: int, is_active: bool, acting_user_id: int):
    """Enable/disable an account; protect self-lockout and the last active admin."""
    if int(user_id) == int(acting_user_id) and not is_active:
        raise ValueError("نمی‌توانید حسابی را که اکنون با آن وارد شده‌اید غیرفعال کنید.")

    connection = get_connection()
    try:
        row = connection.execute(
            "SELECT role, is_active FROM app_users WHERE user_id = ?",
            (int(user_id),),
        ).fetchone()
        if row is None:
            raise ValueError("حساب کاربری پیدا نشد.")

        if row["role"] == "admin" and row["is_active"] and not is_active:
            active_admins = connection.execute(
                "SELECT COUNT(*) FROM app_users WHERE role='admin' AND is_active=1"
            ).fetchone()[0]
            if active_admins <= 1:
                raise ValueError("حداقل یک مدیر فعال باید در سامانه باقی بماند.")

        connection.execute(
            "UPDATE app_users SET is_active = ? WHERE user_id = ?",
            (1 if is_active else 0, int(user_id)),
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def change_user_role(user_id: int, role: str):
    """Change a user's role while ensuring an active admin always remains."""
    if role not in {"admin", "deputy", "teacher"}:
        raise ValueError("نقش کاربر معتبر نیست.")

    connection = get_connection()
    try:
        row = connection.execute(
            "SELECT role, is_active FROM app_users WHERE user_id = ?",
            (int(user_id),),
        ).fetchone()
        if row is None:
            raise ValueError("حساب کاربری پیدا نشد.")

        if row["role"] == "admin" and row["is_active"] and role != "admin":
            active_admins = connection.execute(
                "SELECT COUNT(*) FROM app_users WHERE role='admin' AND is_active=1"
            ).fetchone()[0]
            if active_admins <= 1:
                raise ValueError("نقش آخرین مدیر فعال قابل تغییر نیست.")

        connection.execute(
            "UPDATE app_users SET role = ? WHERE user_id = ?",
            (role, int(user_id)),
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def reset_user_password(user_id: int, new_password: str, force_change: bool = True):
    """Admin resets a password; temporary resets require change at next login."""
    if len(new_password) < 10:
        raise ValueError("رمز عبور باید حداقل ۱۰ نویسه داشته باشد.")

    salt = secrets.token_hex(16)
    password_hash = _hash_password(new_password, salt)
    connection = get_connection()
    try:
        cursor = connection.execute(
            "UPDATE app_users SET password_hash = ?, password_salt = ?, "
            "must_change_password = ?, password_changed_at = CURRENT_TIMESTAMP "
            "WHERE user_id = ?",
            (password_hash, salt, 1 if force_change else 0, int(user_id)),
        )
        if cursor.rowcount == 0:
            raise ValueError("حساب کاربری پیدا نشد.")
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def generate_temporary_password(user_id: int) -> str:
    """Generate a strong one-time-displayed password and require a change at next login."""
    alphabet = string.ascii_letters + string.digits + "!@#$%*-_"
    while True:
        password = "".join(secrets.choice(alphabet) for _ in range(16))
        if (any(c.islower() for c in password)
                and any(c.isupper() for c in password)
                and any(c.isdigit() for c in password)
                and any(c in "!@#$%*-_" for c in password)):
            break
    reset_user_password(user_id, password, force_change=True)
    return password


def change_own_password(user_id: int, new_password: str):
    """Complete forced password change for the currently authenticated user."""
    reset_user_password(user_id, new_password, force_change=False)

def initialize_access_tables():
    """Create per-teacher class/subject assignment storage."""
    connection = get_connection()
    try:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS teacher_assignments (
                assignment_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                class_name TEXT NOT NULL,
                subject_id INTEGER NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES app_users(user_id) ON DELETE CASCADE,
                FOREIGN KEY(subject_id) REFERENCES subjects(subject_id) ON DELETE CASCADE,
                UNIQUE(user_id, class_name, subject_id)
            )
        """)
        connection.commit()
    finally:
        connection.close()


def get_teacher_assignments(user_id):
    connection = get_connection()
    try:
        rows = connection.execute("""
            SELECT ta.class_name, ta.subject_id, s.subject_name
            FROM teacher_assignments ta
            JOIN subjects s ON s.subject_id=ta.subject_id
            WHERE ta.user_id=? ORDER BY ta.class_name, s.subject_name
        """, (int(user_id),)).fetchall()
        return [dict(r) for r in rows]
    finally:
        connection.close()


def set_teacher_assignments(user_id, assignments):
    """Replace assigned class/subject pairs atomically."""
    clean = {(str(c).strip(), int(s)) for c, s in assignments if str(c).strip()}
    connection = get_connection()
    try:
        user = connection.execute(
            "SELECT role FROM app_users WHERE user_id=?", (int(user_id),)
        ).fetchone()
        if not user or user["role"] != "teacher":
            raise ValueError("تخصیص کلاس و درس فقط برای هنرآموز امکان‌پذیر است.")
        for class_name, subject_id in clean:
            class_ok = connection.execute(
                "SELECT 1 FROM students WHERE class_name=? LIMIT 1", (class_name,)
            ).fetchone()
            subject_ok = connection.execute(
                "SELECT 1 FROM subjects WHERE subject_id=?", (subject_id,)
            ).fetchone()
            if not class_ok or not subject_ok:
                raise ValueError("کلاس یا درس انتخاب‌شده معتبر نیست.")
        connection.execute("DELETE FROM teacher_assignments WHERE user_id=?", (int(user_id),))
        connection.executemany(
            "INSERT INTO teacher_assignments(user_id,class_name,subject_id) VALUES(?,?,?)",
            [(int(user_id), c, s) for c, s in sorted(clean)]
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def teacher_scope(user_id):
    rows = get_teacher_assignments(user_id)
    return ({r["class_name"] for r in rows}, {int(r["subject_id"]) for r in rows})


def user_can_access_student(user, student_id, subject_id=None):
    """Check student class and optionally assigned subject at operation time."""
    if user.get("role") in {"admin", "deputy"}:
        return True
    if user.get("role") != "teacher":
        return False
    connection = get_connection()
    try:
        row = connection.execute(
            "SELECT class_name FROM students WHERE student_id=?", (str(student_id),)
        ).fetchone()
        if not row:
            return False
        if subject_id is None:
            allowed = connection.execute(
                "SELECT 1 FROM teacher_assignments WHERE user_id=? AND class_name=? LIMIT 1",
                (int(user["user_id"]), row["class_name"])
            ).fetchone()
        else:
            allowed = connection.execute(
                """SELECT 1 FROM teacher_assignments
                   WHERE user_id=? AND class_name=? AND subject_id=? LIMIT 1""",
                (int(user["user_id"]), row["class_name"], int(subject_id))
            ).fetchone()
        return bool(allowed)
    finally:
        connection.close()


def get_accessible_students(user):
    connection = get_connection()
    try:
        if user.get("role") in {"admin", "deputy"}:
            rows = connection.execute("""
                SELECT student_id, first_name, last_name, grade, major, class_name
                FROM students ORDER BY class_name,last_name,first_name
            """).fetchall()
        elif user.get("role") == "teacher":
            classes, _ = teacher_scope(user["user_id"])
            if not classes:
                return []
            marks = ",".join("?" for _ in classes)
            rows = connection.execute(
                f"""SELECT student_id, first_name, last_name, grade, major, class_name
                    FROM students WHERE class_name IN ({marks})
                    ORDER BY class_name,last_name,first_name""",
                tuple(sorted(classes))
            ).fetchall()
        else:
            return []
        return [dict(r) for r in rows]
    finally:
        connection.close()


def get_accessible_subjects(user):
    connection = get_connection()
    try:
        if user.get("role") in {"admin", "deputy"}:
            rows = connection.execute(
                "SELECT subject_id, subject_name FROM subjects ORDER BY subject_name"
            ).fetchall()
        elif user.get("role") == "teacher":
            _, subjects = teacher_scope(user["user_id"])
            if not subjects:
                return []
            marks = ",".join("?" for _ in subjects)
            rows = connection.execute(
                f"SELECT subject_id, subject_name FROM subjects WHERE subject_id IN ({marks}) ORDER BY subject_name",
                tuple(sorted(subjects))
            ).fetchall()
        else:
            return []
        return [dict(r) for r in rows]
    finally:
        connection.close()

