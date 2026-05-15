# 🚗 Driver Behaviour Analyser

An AI-powered driver behaviour analysis tool that detects traffic signs 
in dashcam footage and provides real-time voice alerts along with a 
driver safety score.

---

## 📌 What It Does

- Uploads a dashcam video
- Detects traffic signs frame by frame using a custom trained YOLO model
- Gives **real-time voice alerts** when a new sign is detected
- Analyses vehicle motion using **Optical Flow**
- Generates a **Driver Safety Score** out of 100
- Lists all **violations** committed during the drive

---

## 🧠 How It Works

Dashcam Video → YOLO Detection → Optical Flow Motion Analysis
→ Voice Alert → Safety Score + Violations Report

---

## 🏗️ Tech Stack

| Tool | Purpose |
|---|---|
| YOLOv8 | Traffic sign detection |
| OpenCV | Video processing and optical flow |
| pyttsx3 | Offline text to speech voice alerts |
| Streamlit | Web UI |
| Python | Core language |

---

## 🚦 Detected Signs

- 🔴 Red Light
- 🟢 Green Light
- 🛑 Stop Sign
- 🔢 Speed Limits: 20, 30, 40, 50, 60, 70, 80, 90, 100

---

## 📊 Scoring System

| Violation | Score Deduction |
|---|---|
| Did not stop at Stop Sign | -30 |
| Ran a Red Light | -40 |
| Possible Speeding | -10 |

| Score | Verdict |
|---|---|
| 80 - 100 | ✅ Good Driver |
| 50 - 79 | ⚠️ Needs Improvement |
| 0 - 49 | 🚨 Dangerous Driving |

---

## ⚙️ Installation

**1. Clone the repository**
```bash
git clone https://github.com/yourusername/driver-behaviour-analyser.git
cd driver-behaviour-analyser
```

**2. Create a virtual environment**
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Add your YOLO model**

Place your trained `best.pt` file in the project root folder or set 
the path via environment variable:
```bash
set MODEL_PATH=C:/path/to/your/best.pt  # Windows
export MODEL_PATH=/path/to/your/best.pt  # Mac/Linux
```

**5. Run the app**
```bash
streamlit run App.py
```

---

## ⚠️ Limitations

- Motion analysis uses optical flow which estimates relative movement 
  — it does not measure actual vehicle speed in km/h
- Model performance depends on training data — signs that look 
  different from training data may not be detected
- Processing speed depends on your laptop hardware

---

## 🔮 Future Improvements

- Live webcam support for real time in-car use
- GPS integration for actual speed measurement
- Mobile app version
- Support for more sign types

---

## 👤 Author

Built by Biveshsingh Sirkissoon.



