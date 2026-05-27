# 🎓 Face Recognition Based Attendance Monitoring System

A real-time, webcam-powered attendance system built with **Python**, **OpenCV**, and **Tkinter**. It automates student attendance by recognizing faces using the LBPH (Local Binary Pattern Histogram) algorithm — no manual roll calls needed.

---

## 📸 How It Works

1. **Register** a new student by capturing 100 face samples via webcam.
2. **Train** the model to learn and store the student's facial profile.
3. **Take Attendance** — the system recognizes faces in real time and logs attendance with date and time to a CSV file.

---

## ✨ Features

- 🔍 Real-time face detection using Haar Cascade Classifier
- 🧠 Face recognition using OpenCV's LBPH Face Recognizer
- 🖥️ Clean GUI built with Tkinter
- 📋 Attendance logs saved as date-stamped CSV files
- 🔐 Password-protected model training
- 🚫 Duplicate attendance prevention (within and across sessions)
- 🕐 Live clock and date display in the UI
- ✅ Supports multiple students

---

## 🗂️ Project Structure

```
Face-Recognition-Based-Attendance-Monitoring-System/
│
├── main.py                          # Main application
├── haarcascade_frontalface_default.xml  # Face detection model
│
├── TrainingImage/                   # Captured face images (auto-created)
├── TrainingImageLabel/              # Trained model + password file (auto-created)
├── StudentDetails/                  # Student CSV registry (auto-created)
└── Attendance/                      # Daily attendance CSV logs (auto-created)
```

---

## 🛠️ Requirements

- Python 3.7+
- A working webcam

### Python Libraries

Install all dependencies with:

```bash
pip install opencv-python opencv-contrib-python numpy Pillow pandas
```

Or use the provided install commands:

```bash
pip install tk
pip install opencv-python
pip install opencv-contrib-python
pip install numpy
pip install Pillow
pip install pandas
```

> ⚠️ `opencv-contrib-python` is required for the LBPH face recognizer (`cv2.face`).

---

## 🚀 Getting Started

**1. Clone the repository**

```bash
git clone https://github.com/datsaryan/Face-Recognition-Based-Attendance-Monitoring-System.git
cd Face-Recognition-Based-Attendance-Monitoring-System
```

**2. Install dependencies**

```bash
pip install opencv-python opencv-contrib-python numpy Pillow pandas
```

**3. Run the application**

```bash
python main.py
```

---

## 📖 Usage Guide

### Registering a New Student

1. Enter the **Student ID** (numbers only) and **Student Name** (letters only) in the right panel.
2. Click **"Take Images"** — the webcam will open and capture up to 100 face samples.
3. Face the camera clearly. A blue rectangle will appear around detected faces.
4. Press **Q** or wait until 100 samples are captured.

### Training the Model

1. After capturing images, click **"Save Profile (Train)"**.
2. Enter the password when prompted (you'll be asked to set one on first use).
3. The system trains the LBPH recognizer and saves the model to `TrainingImageLabel/Trainner.yml`.

### Taking Attendance

1. Click **"Take Attendance"** in the left panel.
2. The webcam opens and begins recognizing faces in real time.
3. Recognized students are highlighted in **green**; unknown faces in **red**.
4. Press **Q** to finish — attendance is saved to `Attendance/Attendance_DD-MM-YYYY.csv`.
5. The Treeview table in the UI updates with today's attendance records.

### Changing the Password

Go to **Help → Change Password** from the menu bar.

---

## 📁 Output Files

| File | Description |
|------|-------------|
| `StudentDetails/StudentDetails.csv` | Registry of all registered students |
| `TrainingImageLabel/Trainner.yml` | Trained LBPH face recognition model |
| `Attendance/Attendance_DD-MM-YYYY.csv` | Daily attendance log with ID, Name, Date, Time |

---

## ⚠️ Known Limitations

- Works best in **well-lit environments**
- Accuracy may drop with glasses, masks, or extreme angles
- Designed for single-camera setups
- macOS users: `cv2.imshow` must run on the main thread (already handled in the code)

---

## 🧰 Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core language |
| OpenCV | Face detection & recognition |
| Tkinter | GUI |
| Pillow (PIL) | Image processing for training |
| NumPy | Array operations |
| Pandas | CSV handling |

---

## 👤 Author

**Aryan** — [@datsaryan](https://github.com/datsaryan)

📧 aryansobdh@gmail.com

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
