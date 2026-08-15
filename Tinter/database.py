import sqlite3

def connect_db():
    return sqlite3.connect("attendance.db")

def create_tables():
    conn = connect_db()
    cur = conn.cursor()

    # Base table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS students(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT
    )
    """)

    # 🔥 Auto add new columns if missing
    try:
        cur.execute("ALTER TABLE students ADD COLUMN gender TEXT")
    except:
        pass

    try:
        cur.execute("ALTER TABLE students ADD COLUMN course TEXT")
    except:
        pass

    try:
        cur.execute("ALTER TABLE students ADD COLUMN section TEXT")
    except:
        pass

    conn.commit()
    conn.close()


def add_student(name, gender, course, section):
    conn = connect_db()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO students(name,gender,course,section) VALUES(?,?,?,?)",
        (name, gender, course, section)
    )
    conn.commit()
    sid = cur.lastrowid
    conn.close()
    return sid


def get_students():
    conn = connect_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM students")
    data = cur.fetchall()
    conn.close()
    return data


def delete_student(sid):
    conn = connect_db()
    cur = conn.cursor()
    cur.execute("DELETE FROM students WHERE id=?", (sid,))
    conn.commit()
    conn.close()


def get_name(sid):
    conn = connect_db()
    cur = conn.cursor()
    cur.execute("SELECT name FROM students WHERE id=?", (sid,))
    data = cur.fetchone()
    conn.close()
    return data[0] if data else "Unknown"