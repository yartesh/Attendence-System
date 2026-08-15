import tkinter as tk
from tkinter import messagebox, ttk, filedialog
from PIL import Image, ImageTk
import os
import csv
import threading
from datetime import datetime

from database import *
from face_system import *

create_tables()

root = tk.Tk()
root.title("Attendance System")
root.geometry("900x600")

def bring_to_front(win):
    """Force any Toplevel window to appear in front of everything."""
    win.lift()
    win.focus_force()
    win.attributes("-topmost", True)
    # Turn topmost off after 300 ms so it doesn't stay glued above all apps forever
    win.after(300, lambda: win.attributes("-topmost", False))

def clear():
    for w in root.winfo_children():
        w.destroy()

def set_bg():
    if os.path.exists("images/bg.jpg"):
        img = Image.open("images/bg.jpg").resize((900, 600))
    else:
        img = Image.new("RGB", (900, 600), "black")

    bg = ImageTk.PhotoImage(img)
    label = tk.Label(root, image=bg)
    label.image = bg
    label.place(x=0, y=0)

def title(txt):
    tk.Label(root, text=txt, font=("Arial", 20, "bold"),
             bg="white", fg="red").pack(fill="x")

def img_btn(path, text, x, y, cmd):
    if os.path.exists(path):
        img = Image.open(path).resize((120, 120))
    else:
        img = Image.new("RGB", (120, 120), "gray")

    photo = ImageTk.PhotoImage(img)
    b = tk.Button(root, image=photo, command=cmd)
    b.image = photo
    b.place(x=x, y=y)

    tk.Label(root, text=text, bg="blue", fg="white").place(x=x, y=y+120, width=120)

# ── HOME ──────────────────────────────────────────────────────────────────────
def home():
    clear()
    set_bg()
    title("FACE RECOGNITION ATTENDANCE SYSTEM")

    img_btn("images/attendance.jpg", "Attendance", 200, 200, start_attendance_session)
    img_btn("images/admin.jpg",      "Admin Login", 380, 200, admin_login)
    img_btn("images/help.jpg",       "Help",        560, 200, help_page)

# ── ATTENDANCE SESSION (camera inside Tkinter — always stays in front) ─────────
def start_attendance_session():
    if not os.path.exists("trainer.yml"):
        messagebox.showerror("Error", "No trained model found!\nPlease ask Admin to add & train students first.")
        return

    # ── Setup recognizer & detector once ──────────────────────────────────────
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer.yml')
    detector   = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        messagebox.showerror("Error", "Camera not found!\nCheck your webcam and try again.")
        return

    # ── Tkinter window for camera feed ────────────────────────────────────────
    win = tk.Toplevel(root)
    win.title("Face Recognition — Attendance")
    win.resizable(False, False)
    bring_to_front(win)

    # Canvas shows the live camera feed
    CAM_W, CAM_H = 640, 400
    canvas = tk.Canvas(win, width=CAM_W, height=CAM_H, bg="black", highlightthickness=0)
    canvas.pack()

    # Status bar below the feed
    status_var = tk.StringVar(value="📷  Looking for face...")
    status_lbl = tk.Label(win, textvariable=status_var,
                          font=("Arial", 11, "bold"),
                          bg="#1a237e", fg="white", pady=6)
    status_lbl.pack(fill="x")

    # Close button
    close_btn = tk.Button(win, text="❌  Close Camera",
                          font=("Arial", 10, "bold"),
                          bg="#e53935", fg="white", pady=5)
    close_btn.pack(fill="x")

    # ── State ─────────────────────────────────────────────────────────────────
    state = {
        "running":     True,
        "marked_ids":  set(),
        "close_after": None,   # datetime when attendance was newly marked
        "photo_ref":   None,   # keep PhotoImage alive
    }

    def stop():
        state["running"] = False
        cam.release()
        try:
            win.destroy()
        except Exception:
            pass

    close_btn.config(command=stop)
    win.protocol("WM_DELETE_WINDOW", stop)

    # ── Per-frame update loop via root.after() — runs on main thread ──────────
    def update():
        if not state["running"]:
            return

        ret, frame = cam.read()
        if not ret:
            stop()
            return

        frame = cv2.resize(frame, (CAM_W, CAM_H))
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.2, 5)

        new_status = "📷  Looking for face..."

        for (x, y, w, h) in faces:
            sid, conf = recognizer.predict(gray[y:y+h, x:x+w])

            if conf < 60:
                name   = get_name(sid)
                result = mark_attendance(sid)

                if result == "marked":
                    if sid not in state["marked_ids"]:
                        state["marked_ids"].add(sid)
                        state["close_after"] = datetime.now()
                    color      = (0, 255, 0)
                    new_status = f"✅  Attendance Marked: {name}  — Closing..."

                elif result == "already":
                    color      = (0, 200, 255)
                    new_status = f"⚠️  Already Marked Today: {name}"
                    cv2.putText(frame,
                                f"Already Marked: {name}",
                                (x, y - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                                (0, 200, 255), 2)
            else:
                name       = "Unknown"
                color      = (0, 0, 255)
                new_status = "❓  Face not recognized"

            cv2.putText(frame, name, (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        # Bottom hint bar on the frame
        bar_y = CAM_H - 30
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, bar_y), (CAM_W, CAM_H), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        cv2.putText(frame, "Click 'Close Camera' button to exit",
                    (10, CAM_H - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 220, 255), 1)

        # Success overlay
        if state["close_after"] is not None:
            cv2.putText(frame, "Attendance Marked!",
                        (CAM_W // 2 - 150, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                        (0, 255, 0), 3)

        # Convert BGR → RGB → PhotoImage and show on canvas
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil   = Image.fromarray(rgb)
        photo = ImageTk.PhotoImage(pil)
        canvas.create_image(0, 0, anchor="nw", image=photo)
        state["photo_ref"] = photo   # prevent garbage collection

        status_var.set(new_status)

        # Auto-close 2 s after first successful mark
        if state["close_after"] is not None:
            elapsed = (datetime.now() - state["close_after"]).total_seconds()
            if elapsed >= 2:
                stop()
                return

        # Schedule next frame (~30 fps)
        win.after(33, update)

    # Start the loop
    win.after(33, update)

# ── LOGIN ─────────────────────────────────────────────────────────────────────
def admin_login():
    win = tk.Toplevel(root)
    bring_to_front(win)

    tk.Label(win, text="Username").pack()
    u = tk.Entry(win); u.pack()

    tk.Label(win, text="Password").pack()
    p = tk.Entry(win, show="*"); p.pack()

    def login():
        if u.get() == "admin" and p.get() == "1234":
            win.destroy()
            admin_panel()
        else:
            messagebox.showerror("Error", "Wrong login")

    tk.Button(win, text="Login", command=login).pack()

# ── ADMIN PANEL ───────────────────────────────────────────────────────────────
def admin_panel():
    clear()
    set_bg()
    title("ADMIN PANEL")

    img_btn("images/add.jpg",  "Add Student",    200, 200, add_student_ui)
    img_btn("images/show.jpg", "Show Students",  380, 200, show_students)
    img_btn("images/data.jpg", "Attendance Data",560, 200, attendance_data)

    tk.Button(root, text="Logout", bg="red", fg="white",
              command=home).place(x=400, y=400)

# ── ADD STUDENT ───────────────────────────────────────────────────────────────
def add_student_ui():
    win = tk.Toplevel(root)
    bring_to_front(win)

    tk.Label(win, text="Name").grid(row=0, column=0)
    tk.Label(win, text="Gender").grid(row=1, column=0)
    tk.Label(win, text="Course").grid(row=2, column=0)
    tk.Label(win, text="Section").grid(row=3, column=0)

    name    = tk.Entry(win); name.grid(row=0, column=1)
    gender  = tk.Entry(win); gender.grid(row=1, column=1)
    course  = tk.Entry(win); course.grid(row=2, column=1)
    section = tk.Entry(win); section.grid(row=3, column=1)

    sid_holder = {"id": None}

    def save():
        sid = add_student(name.get(), gender.get(), course.get(), section.get())
        sid_holder["id"] = sid
        messagebox.showinfo("Saved", "Now Capture Face")

    def capture():
        if sid_holder["id"] is None:
            messagebox.showerror("Error", "Save first")
            return
        capture_images(sid_holder["id"])
        train_model()
        messagebox.showinfo("Done", "Face Added")

    tk.Button(win, text="Save Data",    command=save).grid(row=4, column=0)
    tk.Button(win, text="Capture Face", command=capture).grid(row=4, column=1)

# ── SHOW STUDENTS ─────────────────────────────────────────────────────────────
def show_students():
    win = tk.Toplevel(root)
    bring_to_front(win)

    def load():
        for w in win.winfo_children():
            w.destroy()

        students = get_students()

        if not students:
            tk.Label(win, text="No Students").pack()
            return

        for s in students:
            frame = tk.Frame(win)
            frame.pack(fill="x")

            tk.Label(frame,
                     text=f"{s[0]} | {s[1]} | {s[2]} | {s[3]} | {s[4]}").pack(side="left")

            def delete(sid=s[0]):
                # 🔴 OLD CODE (same)
                delete_student(sid)

                # 🟢 NEW LOGIC (ONLY THIS ADDED)
                if os.path.exists("dataset"):
                    for file in os.listdir("dataset"):
                        if file.startswith(f"User.{sid}."):
                            try:
                                os.remove(os.path.join("dataset", file))
                            except:
                                pass

                messagebox.showinfo("Deleted", "Student deleted")
                load()

            tk.Button(frame, text="Delete", bg="red", fg="white",
                      command=delete).pack(side="right")

    load()
# ── ATTENDANCE DATA (Search, Sort, Export, Delete) ────────────────────────────
def attendance_data():
    win = tk.Toplevel(root)
    win.title("Attendance Data")
    win.geometry("780x520")
    bring_to_front(win)

    # ── Load all rows from CSV ────────────────────────────────────────────────
    def load_rows():
        rows = []
        if not os.path.isfile("attendance.csv"):
            return rows
        with open("attendance.csv", "r") as f:
            reader = csv.reader(f)
            next(reader, None)          # skip header
            for row in reader:
                if len(row) >= 4:
                    rows.append(row)
        return rows

    # ── Save all rows back to CSV (used after delete) ─────────────────────────
    def save_rows(rows):
        with open("attendance.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ID", "Name", "Date", "Time"])
            writer.writerows(rows)

    all_rows = load_rows()

    # ── Search + Sort bar ─────────────────────────────────────────────────────
    top = tk.Frame(win)
    top.pack(fill="x", padx=8, pady=6)

    tk.Label(top, text="Search:").pack(side="left")
    search_var = tk.StringVar()
    tk.Entry(top, textvariable=search_var, width=20).pack(side="left", padx=(2, 10))

    tk.Label(top, text="Sort by:").pack(side="left")
    sort_col = tk.StringVar(value="Date")
    sort_menu = ttk.Combobox(top, textvariable=sort_col,
                             values=["ID", "Name", "Date", "Time"],
                             state="readonly", width=8)
    sort_menu.pack(side="left", padx=(2, 4))

    sort_order = tk.StringVar(value="Ascending")
    order_menu = ttk.Combobox(top, textvariable=sort_order,
                              values=["Ascending", "Descending"],
                              state="readonly", width=10)
    order_menu.pack(side="left", padx=(0, 10))

    # ── Treeview table ────────────────────────────────────────────────────────
    frame_table = tk.Frame(win)
    frame_table.pack(fill="both", expand=True, padx=8, pady=(0, 4))

    cols = ("ID", "Name", "Date", "Time")
    tree = ttk.Treeview(frame_table, columns=cols, show="headings", selectmode="extended")

    for c in cols:
        tree.heading(c, text=c)
        tree.column(c, width=160, anchor="center")

    vsb = ttk.Scrollbar(frame_table, orient="vertical",   command=tree.yview)
    hsb = ttk.Scrollbar(frame_table, orient="horizontal", command=tree.xview)
    tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
    vsb.pack(side="right",  fill="y")
    hsb.pack(side="bottom", fill="x")
    tree.pack(fill="both",  expand=True)

    # ── Status label ──────────────────────────────────────────────────────────
    status_var = tk.StringVar()
    tk.Label(win, textvariable=status_var, anchor="w").pack(fill="x", padx=8)

    # ── Render: filter → sort → populate tree ────────────────────────────────
    col_index = {"ID": 0, "Name": 1, "Date": 2, "Time": 3}

    def render(*_):
        query    = search_var.get().strip().lower()
        col      = col_index[sort_col.get()]
        rev      = (sort_order.get() == "Descending")
        filtered = ([r for r in all_rows
                     if any(query in cell.lower() for cell in r)]
                    if query else list(all_rows))
        filtered.sort(key=lambda r: r[col], reverse=rev)

        tree.delete(*tree.get_children())
        for row in filtered:
            tree.insert("", "end", values=row[:4])

        status_var.set(f"Showing {len(filtered)} of {len(all_rows)} record(s)  |  "
                       f"Click a row to select, then press Delete Record")

    search_var.trace_add("write", render)
    sort_menu.bind("<<ComboboxSelected>>",  render)
    order_menu.bind("<<ComboboxSelected>>", render)
    render()

    # ── Bottom buttons ────────────────────────────────────────────────────────
    btn_frame = tk.Frame(win)
    btn_frame.pack(fill="x", padx=8, pady=6)

    # Refresh
    def refresh():
        nonlocal all_rows
        all_rows = load_rows()
        render()

    # Export visible rows to CSV
    def export_csv():
        dest = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Export Attendance As"
        )
        if not dest:
            return

        query    = search_var.get().strip().lower()
        col      = col_index[sort_col.get()]
        rev      = (sort_order.get() == "Descending")
        filtered = ([r for r in all_rows
                     if any(query in cell.lower() for cell in r)]
                    if query else list(all_rows))
        filtered.sort(key=lambda r: r[col], reverse=rev)

        with open(dest, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ID", "Name", "Date", "Time"])
            writer.writerows(filtered)

        messagebox.showinfo("Exported",
                            f"Exported {len(filtered)} record(s) to:\n{dest}")

    # ── Delete selected rows ──────────────────────────────────────────────────
    def delete_selected():
        selected = tree.selection()
        if not selected:
            messagebox.showwarning("No Selection",
                                   "Please select one or more rows to delete.")
            return

        # Collect the values of selected rows
        to_delete = []
        for item in selected:
            vals = tree.item(item, "values")
            to_delete.append(list(vals))   # [ID, Name, Date, Time]

        # Confirm
        names_str = "\n".join(
            f"  • {r[1]}  ({r[2]})" for r in to_delete
        )
        confirm = messagebox.askyesno(
            "Confirm Delete",
            f"Delete {len(to_delete)} attendance record(s)?\n\n{names_str}"
        )
        if not confirm:
            return

        # Remove matched rows from all_rows
        # Match on ID + Date (unique per person per day)
        delete_keys = {(r[0], r[2]) for r in to_delete}  # (ID, Date)

        nonlocal all_rows
        all_rows = [r for r in all_rows
                    if (r[0], r[2]) not in delete_keys]

        # Write back to CSV
        save_rows(all_rows)

        render()
        messagebox.showinfo("Deleted",
                            f"{len(to_delete)} record(s) deleted successfully.")

    tk.Button(btn_frame, text="🔄 Refresh",
              command=refresh,
              bg="#2196F3", fg="white", width=12).pack(side="left", padx=4)

    tk.Button(btn_frame, text="📤 Export CSV",
              command=export_csv,
              bg="#4CAF50", fg="white", width=12).pack(side="left", padx=4)

    tk.Button(btn_frame, text="🗑 Delete Record",
              command=delete_selected,
              bg="#FF5722", fg="white", width=14).pack(side="left", padx=4)

    tk.Button(btn_frame, text="Close",
              command=win.destroy,
              bg="#f44336", fg="white", width=10).pack(side="right", padx=4)

# ── HELP ──────────────────────────────────────────────────────────────────────
def help_page():
    win = tk.Toplevel(root)
    win.title("Help & Instructions")
    win.geometry("560x620")
    win.resizable(False, False)
    win.configure(bg="#f0f4f8")
    bring_to_front(win)

    # ── Scrollable canvas setup ───────────────────────────────────────────────
    canvas = tk.Canvas(win, bg="#f0f4f8", highlightthickness=0)
    scrollbar = ttk.Scrollbar(win, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)

    scrollbar.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)

    inner = tk.Frame(canvas, bg="#f0f4f8")
    inner_id = canvas.create_window((0, 0), window=inner, anchor="nw")

    def on_configure(e):
        canvas.configure(scrollregion=canvas.bbox("all"))
        canvas.itemconfig(inner_id, width=canvas.winfo_width())

    inner.bind("<Configure>", on_configure)
    canvas.bind("<Configure>", lambda e: canvas.itemconfig(inner_id, width=e.width))

    # Mouse wheel scroll
    def on_mousewheel(e):
        canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")
    canvas.bind_all("<MouseWheel>", on_mousewheel)
    win.bind("<Destroy>", lambda e: canvas.unbind_all("<MouseWheel>"))

    # ── Helper to add sections ────────────────────────────────────────────────
    def section_header(icon, text, color):
        f = tk.Frame(inner, bg=color, pady=6)
        f.pack(fill="x", padx=14, pady=(14, 2))
        tk.Label(f, text=f"  {icon}  {text}", font=("Arial", 12, "bold"),
                 bg=color, fg="white", anchor="w").pack(fill="x", padx=6)

    def bullet(icon, text, indent=20):
        f = tk.Frame(inner, bg="#f0f4f8")
        f.pack(fill="x", padx=indent + 14, pady=2)
        tk.Label(f, text=icon, font=("Arial", 11),
                 bg="#f0f4f8", fg="#444").pack(side="left")
        tk.Label(f, text=f"  {text}", font=("Arial", 10),
                 bg="#f0f4f8", fg="#333", wraplength=440, justify="left",
                 anchor="w").pack(side="left", fill="x")

    def numbered(n, text):
        f = tk.Frame(inner, bg="#ffffff", relief="flat",
                     highlightbackground="#dde", highlightthickness=1)
        f.pack(fill="x", padx=28, pady=3)
        tk.Label(f, text=f" {n} ", font=("Arial", 10, "bold"),
                 bg="#4a90d9", fg="white", width=3).pack(side="left")
        tk.Label(f, text=f"  {text}", font=("Arial", 10),
                 bg="#ffffff", fg="#222", anchor="w",
                 wraplength=420, justify="left").pack(side="left", fill="x", padx=4, pady=4)

    def spacer(h=6):
        tk.Frame(inner, bg="#f0f4f8", height=h).pack()

    def divider():
        tk.Frame(inner, bg="#ccc", height=1).pack(fill="x", padx=14, pady=4)

    # ── Title ─────────────────────────────────────────────────────────────────
    tk.Label(inner, text="❓  Help & Instructions",
             font=("Arial", 16, "bold"),
             bg="#1a237e", fg="white", pady=12).pack(fill="x")

    spacer(8)

    # ── How to Mark Attendance ────────────────────────────────────────────────
    section_header("📸", "How to Mark Attendance?", "#1976D2")
    spacer(4)
    numbered(1, "Go to the Attendance section from the home page.")
    numbered(2, "Camera will open automatically.")
    numbered(3, "Stand in front of the camera — make sure your face is visible.")
    numbered(4, "Your attendance will be marked automatically.")
    numbered(5, "Camera closes on its own after marking. Press Q or ESC to exit anytime.")

    divider()

    # ── New Student ───────────────────────────────────────────────────────────
    section_header("👤", "New Student?", "#388E3C")
    spacer(4)
    bullet("ℹ️", "If you are a new student, contact the Admin for registration before marking attendance.")
    bullet("ℹ️", "Admin will add your details and capture your face images to train the system.")

    divider()

    # ── Important Instructions ────────────────────────────────────────────────
    section_header("⚠️", "Important Instructions", "#F57C00")
    spacer(4)
    bullet("✔️", "Make sure your face is clearly visible to the camera.")
    bullet("✔️", "Stand properly in front of the camera — do not tilt your face.")
    bullet("✔️", "Ensure good lighting — avoid sitting against a bright background.")
    bullet("✔️", "Attendance can be marked only once per day.")
    bullet("✔️", "If already marked, the camera will show a yellow warning.")

    divider()

    # ── Troubleshooting ───────────────────────────────────────────────────────
    section_header("🛠", "Troubleshooting", "#7B1FA2")
    spacer(4)
    bullet("❌", "Camera not opening → Check your webcam connection and try again.")
    bullet("❌", "Face not detected → Improve lighting and face position.")
    bullet("❌", "Attendance not marked → Move closer and look directly into the camera.")
    bullet("❌", "Showing 'Unknown' → Your face is not registered. Contact Admin.")
    bullet("❌", "Already marked warning → Attendance for today is already recorded.")

    divider()
    spacer(10)

    # ── Home button ───────────────────────────────────────────────────────────
    tk.Button(inner, text="🏠  Back to Home",
              font=("Arial", 11, "bold"),
              bg="#1a237e", fg="white",
              relief="flat", pady=8,
              command=win.destroy).pack(fill="x", padx=40, pady=(4, 18))

# ── START ─────────────────────────────────────────────────────────────────────
home()
root.mainloop()