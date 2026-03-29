"""
Face Recognition Based Attendance Monitoring System
Fixed: cv2.imshow must run on main thread on macOS.
Strategy: hide the Tk window, run the camera loop on the main thread,
then restore the Tk window. Training (no imshow) keeps its background thread.
"""

# ─────────────────────────── IMPORTS ───────────────────────────
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog, font as tkfont
import cv2
import os
import csv
import threading
import numpy as np
from PIL import Image
import pandas as pd
import datetime
import time

# ─────────────────────────── CONSTANTS ─────────────────────────
HAAR_CASCADE   = "haarcascade_frontalface_default.xml"
TRAINING_DIR   = "TrainingImage"
LABEL_DIR      = "TrainingImageLabel"
STUDENT_DIR    = "StudentDetails"
ATTENDANCE_DIR = "Attendance"
STUDENT_CSV    = os.path.join(STUDENT_DIR,  "StudentDetails.csv")
TRAINER_YML    = os.path.join(LABEL_DIR,    "Trainner.yml")
PASSWORD_FILE  = os.path.join(LABEL_DIR,    "psd.txt")

CSV_COLUMNS = ["ID", "NAME"]
ATT_COLUMNS = ["ID", "Name", "Date", "Time"]


# ─────────────────────────── HELPERS ───────────────────────────
def assure_path_exists(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_best_font(root: tk.Tk) -> str:
    """Must be called AFTER tk.Tk() is created."""
    available = tkfont.families(root)
    for candidate in ("Comic Sans MS", "Helvetica Neue", "Helvetica", "Arial"):
        if candidate in available:
            return candidate
    return "TkDefaultFont"


def check_haarcascade() -> bool:
    return os.path.isfile(HAAR_CASCADE)


def read_password():
    if os.path.isfile(PASSWORD_FILE):
        with open(PASSWORD_FILE, "r") as f:
            return f.read().strip()
    return None


def write_password(pw: str) -> None:
    assure_path_exists(LABEL_DIR)
    with open(PASSWORD_FILE, "w") as f:
        f.write(pw)


def get_next_serial() -> int:
    if not os.path.isfile(STUDENT_CSV):
        return 1
    with open(STUDENT_CSV, "r", newline="") as f:
        return max(1, sum(1 for _ in csv.reader(f)) - 1)


def count_registrations() -> int:
    if not os.path.isfile(STUDENT_CSV):
        return 0
    with open(STUDENT_CSV, "r", newline="") as f:
        return max(0, len(list(csv.reader(f))) - 1)


# ─────────────────────────── PASSWORD WINDOW ───────────────────
class ChangePasswordWindow:
    def __init__(self, parent_font: str):
        # Use Toplevel so we don't create a second Tk root
        self.win = tk.Toplevel()
        self.win.title("Change Password")
        self.win.geometry("420x175")
        self.win.resizable(False, False)
        self.win.configure(bg="white")
        self.win.grab_set()
        F = parent_font

        labels = ["Old Password", "New Password", "Confirm New Password"]
        self.entries = []
        for i, lbl in enumerate(labels):
            tk.Label(self.win, text=lbl, bg="white",
                     font=(F, 12, "bold"), width=22, anchor="w").place(x=10, y=10 + i * 42)
            e = tk.Entry(self.win, width=22, fg="black", relief="solid",
                         font=(F, 12, "bold"), show="*")
            e.place(x=210, y=10 + i * 42)
            self.entries.append(e)

        tk.Button(self.win, text="Save", command=self._save,
                  fg="black", bg="#00fcca", width=18,
                  font=(F, 10, "bold")).place(x=10, y=136)
        tk.Button(self.win, text="Cancel", command=self.win.destroy,
                  fg="black", bg="#ff5555", width=18,
                  font=(F, 10, "bold")).place(x=220, y=136)

    def _save(self) -> None:
        old_pw, new_pw, confirm_pw = (e.get() for e in self.entries)
        stored = read_password()

        if stored is None:
            if not new_pw:
                messagebox.showwarning("Empty Password", "Password cannot be empty.")
                return
            if new_pw != confirm_pw:
                messagebox.showerror("Mismatch", "New passwords do not match.")
                return
            write_password(new_pw)
            messagebox.showinfo("Success", "Password set successfully!")
            self.win.destroy()
            return

        if old_pw != stored:
            messagebox.showerror("Wrong Password", "Old password is incorrect.")
            return
        if new_pw != confirm_pw:
            messagebox.showerror("Mismatch", "New passwords do not match.")
            return
        if not new_pw:
            messagebox.showwarning("Empty Password", "Password cannot be empty.")
            return

        write_password(new_pw)
        messagebox.showinfo("Success", "Password changed successfully!")
        self.win.destroy()


# ─────────────────────────── MAIN APPLICATION ──────────────────
class AttendanceApp:
    def __init__(self):
        # tk.Tk() MUST be created before any tkinter font API calls
        self.window = tk.Tk()
        self.window.title("Attendance System")
        self.window.geometry("1280x720")
        self.window.resizable(True, False)
        self.window.configure(bg="#1a2a0a")

        self.FONT = get_best_font(self.window)

        for d in (TRAINING_DIR, LABEL_DIR, STUDENT_DIR, ATTENDANCE_DIR):
            assure_path_exists(d)

        self._build_ui()
        self._refresh_registration_count()
        self._tick()
        self.window.mainloop()

    # ── Clock ────────────────────────────────────────────────────
    def _tick(self) -> None:
        self.clock_label.config(text=time.strftime("%H:%M:%S"))
        self.window.after(500, self._tick)

    # ── Helpers ──────────────────────────────────────────────────
    def _set_status(self, text: str) -> None:
        self.status_label.configure(text=text)

    def _refresh_registration_count(self) -> None:
        self.reg_count_label.configure(
            text=f"Total Registrations : {count_registrations()}"
        )

    def _contact(self) -> None:
        messagebox.showinfo("Contact Us", "For support: aryansobdh@gmail.com")

    # ── UI ───────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        F = self.FONT

        tk.Label(
            self.window,
            text="Face Recognition Based Attendance Monitoring System",
            fg="white", bg="#1a2a0a", font=(F, 24, "bold"),
        ).place(x=10, y=10)

        ts = time.time()
        date = datetime.datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        day, month, year = date.split("-")
        months = {
            "01": "January",  "02": "February", "03": "March",
            "04": "April",    "05": "May",       "06": "June",
            "07": "July",     "08": "August",    "09": "September",
            "10": "October",  "11": "November",  "12": "December",
        }
        date_frame = tk.Frame(self.window, bg="#1a2a0a")
        date_frame.place(relx=0.37, rely=0.09, relwidth=0.16, relheight=0.07)
        tk.Label(date_frame,
                 text=f"{day} {months[month]} {year}  |  ",
                 fg="#ff61e5", bg="#1a2a0a",
                 font=(F, 20, "bold")).pack(fill="both", expand=True)

        clock_frame = tk.Frame(self.window, bg="#1a2a0a")
        clock_frame.place(relx=0.53, rely=0.09, relwidth=0.09, relheight=0.07)
        self.clock_label = tk.Label(clock_frame, fg="#ff61e5", bg="#1a2a0a",
                                    font=(F, 20, "bold"))
        self.clock_label.pack(fill="both", expand=True)

        # Left frame
        self.frame1 = tk.Frame(self.window, bg="#c79cff")
        self.frame1.place(relx=0.11, rely=0.17, relwidth=0.39, relheight=0.80)

        tk.Label(self.frame1, text="Already Registered  -  Take Attendance",
                 fg="black", bg="#00fcca", font=(F, 14, "bold")).place(x=0, y=0)

        tk.Button(
            self.frame1, text="Take Attendance",
            command=self._take_attendance,
            fg="black", bg="#3ffc00", width=35, height=1,
            activebackground="white", font=(F, 14, "bold"),
        ).place(x=30, y=40)

        self.tv = ttk.Treeview(self.frame1, height=13,
                               columns=("name", "date", "time"))
        for col, w, label in [
            ("#0",   82,  "ID"),
            ("name", 130, "NAME"),
            ("date", 133, "DATE"),
            ("time", 133, "TIME"),
        ]:
            self.tv.column(col, width=w)
            self.tv.heading(col, text=label)
        self.tv.grid(row=2, column=0, padx=0, pady=(140, 0), columnspan=4)

        scroll = ttk.Scrollbar(self.frame1, orient="vertical", command=self.tv.yview)
        scroll.grid(row=2, column=4, padx=(0, 80), pady=(140, 0), sticky="ns")
        self.tv.configure(yscrollcommand=scroll.set)

        tk.Button(
            self.frame1, text="Quit", command=self.window.destroy,
            fg="black", bg="#eb4600", width=35, height=1,
            activebackground="white", font=(F, 14, "bold"),
        ).place(x=30, y=430)

        # Right frame
        self.frame2 = tk.Frame(self.window, bg="#c79cff")
        self.frame2.place(relx=0.51, rely=0.17, relwidth=0.38, relheight=0.80)

        tk.Label(self.frame2, text="New Student Registration",
                 fg="black", bg="#00fcca", font=(F, 14, "bold")).grid(row=0, column=0)

        tk.Label(self.frame2, text="Student ID", width=20, fg="black",
                 bg="#c79cff", font=(F, 15, "bold")).place(x=80, y=55)
        self.id_entry = tk.Entry(self.frame2, width=28, fg="white", font=(F, 14, "bold"))
        self.id_entry.place(x=30, y=88)
        tk.Button(self.frame2, text="Clear",
                  command=lambda: self.id_entry.delete(0, "end"),
                  fg="black", bg="#ff7221", width=9,
                  activebackground="white", font=(F, 10, "bold")).place(x=355, y=86)

        tk.Label(self.frame2, text="Student Name", width=20, fg="black",
                 bg="#c79cff", font=(F, 15, "bold")).place(x=80, y=140)
        self.name_entry = tk.Entry(self.frame2, width=28, fg="white", font=(F, 14, "bold"))
        self.name_entry.place(x=30, y=173)
        tk.Button(self.frame2, text="Clear",
                  command=lambda: self.name_entry.delete(0, "end"),
                  fg="black", bg="#ff7221", width=9,
                  activebackground="white", font=(F, 10, "bold")).place(x=355, y=172)

        self.status_label = tk.Label(
            self.frame2, text="1) Take Images   >>   2) Save Profile",
            bg="#c79cff", fg="#1a2a0a", width=39, height=1, font=(F, 13, "bold"),
        )
        self.status_label.place(x=7, y=230)

        self.reg_count_label = tk.Label(
            self.frame2, text="", bg="#c79cff", fg="black",
            width=39, height=1, font=(F, 14, "bold"),
        )
        self.reg_count_label.place(x=7, y=450)

        tk.Button(
            self.frame2, text="Take Images",
            command=self._take_images,
            fg="white", bg="#6d00fc", width=34, height=1,
            activebackground="white", font=(F, 14, "bold"),
        ).place(x=30, y=290)

        tk.Button(
            self.frame2, text="Save Profile (Train)",
            command=self._prompt_password_then_train,
            fg="white", bg="#6d00fc", width=34, height=1,
            activebackground="white", font=(F, 14, "bold"),
        ).place(x=30, y=365)

        menubar  = tk.Menu(self.window, relief="ridge")
        helpmenu = tk.Menu(menubar, tearoff=0)
        helpmenu.add_command(label="Change Password",
                             command=lambda: ChangePasswordWindow(self.FONT))
        helpmenu.add_command(label="Contact Us", command=self._contact)
        helpmenu.add_separator()
        helpmenu.add_command(label="Exit", command=self.window.destroy)
        menubar.add_cascade(label="Help", menu=helpmenu)
        self.window.configure(menu=menubar)

    # ── Take Images ──────────────────────────────────────────────
    # Must run on main thread — cv2.imshow crashes in background threads on macOS
    def _take_images(self) -> None:
        if not check_haarcascade():
            messagebox.showerror("Missing File",
                                 f"'{HAAR_CASCADE}' not found. "
                                 "Please add it to the project directory.")
            return

        student_id   = self.id_entry.get().strip()
        student_name = self.name_entry.get().strip()

        if not student_id.isdigit():
            messagebox.showwarning("Invalid ID", "Student ID must be a number.")
            return
        if not student_name.replace(" ", "").isalpha():
            messagebox.showwarning("Invalid Name",
                                   "Name must contain letters and spaces only.")
            return

        serial = get_next_serial()

        if not os.path.isfile(STUDENT_CSV):
            with open(STUDENT_CSV, "w", newline="") as f:
                csv.writer(f).writerow(CSV_COLUMNS)

        # Temporarily hide Tk so OpenCV window can take focus on macOS
        self.window.withdraw()
        self.window.update()

        cam      = cv2.VideoCapture(0)
        detector = cv2.CascadeClassifier(HAAR_CASCADE)
        sample   = 0

        while True:
            ret, frame = cam.read()
            if not ret or frame is None:
                continue
            frame = cv2.flip(frame, 1)
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                sample += 1
                img_path = os.path.join(
                    TRAINING_DIR,
                    f"{student_name}.{serial}.{student_id}.{sample}.jpg",
                )
                cv2.imwrite(img_path, gray[y:y + h, x:x + w])
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, f"Sample {sample}/100", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("Taking Images  |  Press Q to stop", frame)
            if cv2.waitKey(100) & 0xFF == ord("q") or sample >= 100:
                break

        cam.release()
        cv2.destroyAllWindows()

        self.window.deiconify()
        self.window.lift()

        if sample == 0:
            messagebox.showwarning("No Face Detected",
                                   "No face was captured. Please try again.")
            return

        with open(STUDENT_CSV, "a", newline="") as f:
            csv.writer(f).writerow([student_id, student_name])

        self._set_status(f"Images captured for ID: {student_id} ({sample} samples)")
        self._refresh_registration_count()

    # ── Train (background thread is fine — no imshow) ────────────
    def _prompt_password_then_train(self) -> None:
        stored = read_password()
        if stored is None:
            new_pw = simpledialog.askstring(
                "Set Password", "No password found. Set a new password:", show="*"
            )
            if not new_pw:
                messagebox.showwarning("Cancelled", "Password not set. Please try again.")
                return
            write_password(new_pw)
            messagebox.showinfo("Password Set", "Password registered. Saving profile now...")
            threading.Thread(target=self._train_images, daemon=True).start()
            return

        entered = simpledialog.askstring(
            "Password", "Enter password to save profile:", show="*"
        )
        if entered is None:
            return
        if entered != stored:
            messagebox.showerror("Wrong Password", "Incorrect password.")
            return
        threading.Thread(target=self._train_images, daemon=True).start()

    def _train_images(self) -> None:
        if not check_haarcascade():
            messagebox.showerror("Missing File", f"'{HAAR_CASCADE}' not found.")
            return

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        faces, ids = self._get_images_and_labels(TRAINING_DIR)

        if not faces:
            messagebox.showwarning(
                "No Data", "No training images found. Please register students first."
            )
            return

        try:
            recognizer.train(faces, np.array(ids))
        except cv2.error as e:
            messagebox.showerror("Training Error", str(e))
            return

        recognizer.save(TRAINER_YML)
        self.window.after(0, lambda: self._set_status("Profile saved successfully"))
        self.window.after(0, self._refresh_registration_count)

    @staticmethod
    def _get_images_and_labels(path: str):
        """
        Filename format: <name>.<serial>.<student_id>.<sample>.jpg
        Index [2] is the student ID used for recognition training.
        """
        valid_ext = (".jpg", ".jpeg", ".png")
        image_paths = [
            os.path.join(path, f)
            for f in os.listdir(path)
            if f.lower().endswith(valid_ext) and not f.startswith(".")
        ]
        faces, ids = [], []
        for img_path in image_paths:
            try:
                parts = os.path.splitext(os.path.basename(img_path))[0].split(".")
                student_id = int(parts[2])
            except (IndexError, ValueError):
                continue
            pil_img = Image.open(img_path).convert("L")
            faces.append(np.array(pil_img, dtype="uint8"))
            ids.append(student_id)
        return faces, ids

    # ── Take Attendance ──────────────────────────────────────────
    # Must run on main thread — cv2.imshow crashes in background threads on macOS
    def _take_attendance(self) -> None:
        if not check_haarcascade():
            messagebox.showerror("Missing File", f"'{HAAR_CASCADE}' not found.")
            return
        if not os.path.isfile(TRAINER_YML):
            messagebox.showwarning(
                "No Model", "Trained model not found. Please save a profile first."
            )
            return
        if not os.path.isfile(STUDENT_CSV):
            messagebox.showwarning(
                "No Students",
                "Student details file missing. Please register students first."
            )
            return

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(TRAINER_YML)
        cascade = cv2.CascadeClassifier(HAAR_CASCADE)
        df      = pd.read_csv(STUDENT_CSV)

        self.window.withdraw()
        self.window.update()

        cam  = cv2.VideoCapture(0)
        font = cv2.FONT_HERSHEY_SIMPLEX
        recognized_records = []

        while True:
            ret, frame = cam.read()
            if not ret or frame is None:
                continue
            frame = cv2.flip(frame, 1)
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

            for (x, y, w, h) in faces:
                pred_id, conf = recognizer.predict(gray[y:y + h, x:x + w])
                ts       = time.time()
                date_str = datetime.datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
                time_str = datetime.datetime.fromtimestamp(ts).strftime("%H:%M:%S")

                if conf < 50:
                    match = df[df["ID"] == pred_id]
                    if not match.empty:
                        name      = str(match["NAME"].values[0])
                        sid       = str(pred_id)
                        label_txt = f"{name} ({sid})"
                        recognized_records.append([sid, name, date_str, time_str])
                    else:
                        label_txt = f"ID:{pred_id} (not in DB)"
                else:
                    label_txt = "Unknown"

                color = (0, 200, 0) if conf < 50 else (0, 0, 200)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label_txt, (x, y + h + 20),
                            font, 0.7, (255, 255, 255), 2)

            cv2.imshow("Taking Attendance  |  Press Q to finish", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cam.release()
        cv2.destroyAllWindows()

        self.window.deiconify()
        self.window.lift()

        if not recognized_records:
            messagebox.showwarning(
                "No Attendance", "No face was recognized. Please try again."
            )
            return

        seen = set()
        unique_records = []
        for rec in recognized_records:
            if rec[0] not in seen:
                seen.add(rec[0])
                unique_records.append(rec)

        ts       = time.time()
        date_str = datetime.datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        att_csv  = os.path.join(ATTENDANCE_DIR, f"Attendance_{date_str}.csv")
        write_header = not os.path.isfile(att_csv)

        with open(att_csv, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(ATT_COLUMNS)
            writer.writerows(unique_records)

        self._refresh_treeview(att_csv)

    def _refresh_treeview(self, att_csv: str) -> None:
        for row in self.tv.get_children():
            self.tv.delete(row)
        if not os.path.isfile(att_csv):
            return
        with open(att_csv, "r", newline="") as f:
            rows = list(csv.reader(f))
        for record in rows[1:]:
            if len(record) >= 4:
                self.tv.insert("", 0, text=record[0],
                               values=(record[1], record[2], record[3]))


# ─────────────────────────── ENTRY POINT ───────────────────────
if __name__ == "__main__":
    AttendanceApp()