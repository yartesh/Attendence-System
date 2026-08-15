from flask import Flask, render_template, request, redirect, Response, session
import sqlite3
import cv2
import numpy as np
import os
import shutil
from datetime import datetime

app = Flask(__name__)
app.secret_key = "1234"

# ===== GLOBAL =====
current_student = None
capture_mode = False
capture_count = 0
max_images = 20
capture_done = False

# ===== DATABASE =====
conn = sqlite3.connect("attendance.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("CREATE TABLE IF NOT EXISTS users(username TEXT,password TEXT)")
cursor.execute("""CREATE TABLE IF NOT EXISTS students(
id INTEGER PRIMARY KEY AUTOINCREMENT,
student_id TEXT,name TEXT,gender TEXT,phone TEXT,course TEXT,section TEXT)""")
cursor.execute("""CREATE TABLE IF NOT EXISTS attendance(
id INTEGER PRIMARY KEY AUTOINCREMENT,
name TEXT,date TEXT,time TEXT)""")

cursor.execute("INSERT OR IGNORE INTO users VALUES('admin','1234')")
conn.commit()

# ===== FACE =====
dataset_path = "dataset"
os.makedirs(dataset_path, exist_ok=True)

recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades+'haarcascade_frontalface_default.xml'
)

camera = cv2.VideoCapture(0)

def train():
    faces, labels = [], []
    names = {}
    label_id = 0

    for person in os.listdir(dataset_path):
        path = os.path.join(dataset_path, person)
        if not os.path.isdir(path):
            continue

        names[label_id] = person

        for img in os.listdir(path):
            gray = cv2.imread(os.path.join(path, img), 0)
            if gray is None:
                continue
            faces.append(gray)
            labels.append(label_id)

        label_id += 1

    if len(faces) > 0:
        recognizer.train(faces, np.array(labels))

    return names

names = train()

# ===== CAMERA =====
def gen_frames():
    global capture_mode, capture_count, current_student, names, capture_done

    while True:
        ret, frame = camera.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 3)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

            # ===== CAPTURE =====
            if capture_mode:
                path = os.path.join(dataset_path, current_student)
                os.makedirs(path, exist_ok=True)

                capture_count += 1
                cv2.imwrite(f"{path}/{capture_count}.jpg", face)

                cv2.putText(frame, f"Capturing {capture_count}/{max_images}",
                            (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

                if capture_count >= max_images:
                    capture_mode = False
                    capture_count = 0
                    capture_done = True
                    names = train()

            # ===== RECOGNITION =====
            elif len(names) > 0:
                label, conf = recognizer.predict(face)

                if conf < 70:
                    student_id = names[label]

                    cursor.execute("SELECT name FROM students WHERE student_id=?", (student_id,))
                    result = cursor.fetchone()
                    name = result[0] if result else "Unknown"

                    now = datetime.now()
                    date = now.strftime("%Y-%m-%d")
                    time = now.strftime("%H:%M:%S")

                    cursor.execute("SELECT * FROM attendance WHERE name=? AND date=?", (name, date))
                    data = cursor.fetchone()

                    if data is None:
                        cursor.execute("INSERT INTO attendance VALUES(NULL,?,?,?)", (name, date, time))
                        conn.commit()
                        text = f"{student_id} - {name} ✔"
                    else:
                        text = f"{student_id} - {name} ❌ Already"

                else:
                    text = "Unknown"

                cv2.putText(frame, text, (x,y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            break

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# ===== ROUTES =====
@app.route('/')
def home():
    return render_template("home.html")

@app.route('/attendance')
def attendance():
    return render_template("camera.html")

@app.route('/capture')
def capture():
    return render_template("capture.html")

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ===== CAPTURE CHECK =====
@app.route('/check_capture')
def check_capture():
    global capture_done
    if capture_done:
        return "done"
    return "running"

# ===== LOGIN =====
@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == "POST":
        u = request.form['username']
        p = request.form['password']
        cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (u,p))
        if cursor.fetchone():
            session['admin'] = True
            return redirect('/admin')
    return render_template("login.html")

# ===== LOGOUT (FIXED) =====
@app.route('/logout')
def logout():
    session.clear()
    return redirect('/')

# ===== ADMIN PROTECTION =====
@app.route('/admin')
def admin():
    if not session.get('admin'):
        return redirect('/login')
    return render_template("admin.html")

# ===== REGISTER =====
@app.route('/register', methods=['GET','POST'])
def register():
    global capture_mode, current_student

    if request.method == "POST":
        student_id = request.form['student_id'].strip()
        name = request.form['name'].strip()
        gender = request.form['gender'].strip()
        phone = request.form['phone'].strip()
        course = request.form['course'].strip()
        section = request.form['section'].strip()

        if not student_id or not name or not gender or not phone or not course or not section:
            return "❌ All fields required!"

        cursor.execute("SELECT * FROM students WHERE student_id=?", (student_id,))
        if cursor.fetchone():
            return "❌ ID already exists!"

        cursor.execute("""INSERT INTO students(student_id,name,gender,phone,course,section)
        VALUES(?,?,?,?,?,?)""",(student_id,name,gender,phone,course,section))
        conn.commit()

        current_student = student_id
        capture_mode = True
        return redirect('/capture')

    return render_template("register.html")

# ===== STUDENTS =====
@app.route('/students')
def students():
    cursor.execute("SELECT * FROM students")
    return render_template("students.html", data=cursor.fetchall())

# ===== DELETE =====
@app.route('/delete/<student_id>')
def delete(student_id):

    cursor.execute("SELECT name FROM students WHERE student_id=?", (student_id,))
    result = cursor.fetchone()

    if result:
        name = result[0]
        cursor.execute("DELETE FROM attendance WHERE name=?", (name,))
        conn.commit()

    cursor.execute("DELETE FROM students WHERE student_id=?", (student_id,))
    conn.commit()

    path = os.path.join("dataset", student_id)
    if os.path.exists(path):
        shutil.rmtree(path)

    return redirect('/students')

# ===== REPORT =====
@app.route('/report')
def report():
    date = request.args.get('date')
    student = request.args.get('student')

    query = """
    SELECT students.student_id, students.name, attendance.date, attendance.time
    FROM attendance
    JOIN students ON students.name = attendance.name
    WHERE 1=1
    """

    params = []

    if date:
        query += " AND attendance.date=?"
        params.append(date)

    if student:
        query += " AND (students.student_id=? OR students.name=?)"
        params.append(student)
        params.append(student)

    cursor.execute(query, params)
    return render_template("report.html", data=cursor.fetchall())

# ===== EXPORT =====
@app.route('/export')
def export():
    cursor.execute("""
    SELECT students.student_id, students.name, attendance.date, attendance.time
    FROM attendance
    JOIN students ON students.name = attendance.name
    """)
    data = cursor.fetchall()

    def gen():
        yield "Student ID,Name,Date,Time\n"
        for r in data:
            yield f"{r[0]},{r[1]},{r[2]},{r[3]}\n"

    return Response(gen(), mimetype='text/csv',
        headers={"Content-Disposition":"attachment;filename=attendance.csv"})

@app.route('/help')
def help_page():
    return render_template("help.html")

if __name__ == "__main__":
    app.run(debug=True)