import cv2
import os
import numpy as np
from PIL import Image
from datetime import datetime
import csv
import winsound
from database import get_name


def capture_images(sid):
    cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        print("Camera error")
        return

    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    if not os.path.exists("dataset"):
        os.makedirs("dataset")

    count = 0

    while True:
        ret, img = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            cv2.imwrite(f"dataset/User.{sid}.{count}.jpg", gray[y:y+h, x:x+w])
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        cv2.imshow("Capturing Face", img)

        if cv2.waitKey(1) == 13 or count >= 50:
            break

    cam.release()
    cv2.destroyAllWindows()


def train_model():
    path = 'dataset'
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    faces = []
    ids = []

    for file in os.listdir(path):
        img_path = os.path.join(path, file)
        img = Image.open(img_path).convert('L')
        img_np = np.array(img, 'uint8')

        sid = int(file.split(".")[1])

        faces.append(img_np)
        ids.append(sid)

    recognizer.train(faces, np.array(ids))
    recognizer.save('trainer.yml')


def mark_attendance(sid):
    """
    Marks attendance for the given student ID.
    Returns:
        "marked"  → newly marked right now
        "already" → already marked today
    """
    name = get_name(sid)
    now  = datetime.now()

    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    file_exists = os.path.isfile("attendance.csv")
    already = False

    if file_exists:
        with open("attendance.csv", "r") as f:
            for row in csv.reader(f):
                if len(row) >= 3 and row[0] == str(sid) and row[2] == date:
                    already = True
                    break

    if not already:
        with open("attendance.csv", "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["ID", "Name", "Date", "Time"])
            writer.writerow([sid, name, date, time])

        winsound.Beep(1000, 300)
        return "marked"   # newly marked

    return "already"      # already marked today


def recognize_face(stop_flag=None):
    """
    Runs the face recognition camera loop.
    - Press Q / ESC → close camera manually
    - Press Enter   → close camera manually
    - stop_flag["stop"] = True → close from Tkinter side
    - Auto-closes 2 sec after successful NEW attendance mark

    Overlays shown on camera:
    - Green  "Attendance Marked! Closing..."  → just marked
    - Yellow "Already Marked Today: <Name>"  → already done today
    - Red    "Unknown"                        → not recognised
    - Bottom bar: Press Q or ESC hint
    """
    if not os.path.exists("trainer.yml"):
        print("Train model first!")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer.yml')

    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        print("Camera error")
        return

    marked_ids    = set()   # IDs newly marked this session
    already_ids   = set()   # IDs already marked today (shown in yellow)
    close_after   = None    # timestamp of first successful new mark

    while True:

        # External stop from Tkinter
        if stop_flag is not None and stop_flag.get("stop", False):
            break

        ret, img = cam.read()
        if not ret:
            break

        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.2, 5)

        for (x, y, w, h) in faces:
            sid, conf = recognizer.predict(gray[y:y+h, x:x+w])

            if conf < 60:
                name   = get_name(sid)
                result = mark_attendance(sid)

                if result == "marked":
                    # Newly marked — green, start auto-close timer
                    if sid not in marked_ids:
                        marked_ids.add(sid)
                        close_after = datetime.now()
                    color        = (0, 255, 0)
                    label        = name

                elif result == "already":
                    # Already marked today — yellow warning
                    already_ids.add(sid)
                    color = (0, 200, 255)   # yellow-ish in BGR
                    label = name

                    # Show "Already Marked" banner above the box
                    cv2.putText(img,
                                f"Already Marked Today: {name}",
                                (x, y - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 200, 255), 2)
            else:
                name  = "Unknown"
                color = (0, 0, 255)
                label = name

            cv2.putText(img, label, (x, y - 10), 2, 1, color, 2)
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)

        # ── Bottom bar — always visible ──────────────────────────────────────
        h_img = img.shape[0]
        w_img = img.shape[1]
        bar_h = 36
        overlay = img.copy()
        cv2.rectangle(overlay, (0, h_img - bar_h), (w_img, h_img), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
        cv2.putText(img,
                    "Press Q or ESC to Close Camera",
                    (10, h_img - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0, 220, 255), 2)

        # ── Success overlay (after new mark) ─────────────────────────────────
        if close_after is not None:
            cv2.putText(img, "Attendance Marked! Closing...",
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2)

        cv2.imshow("Face Recognition", img)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:   # Q or ESC
            break
        if key == 13:                       # Enter
            break

        # Auto-close 2 seconds after first successful new mark
        if close_after is not None:
            if (datetime.now() - close_after).total_seconds() >= 2:
                break

    cam.release()
    cv2.destroyAllWindows()