# 🚗 Driver Behaviour Analyser

## 📌 Problem Statement

Road safety remains one of the most critical challenges worldwide. 
A significant number of road accidents occur due to drivers failing 
to obey traffic signs — running red lights, ignoring stop signs, 
and exceeding speed limits. 

Traditional methods of monitoring driver behaviour rely on human 
supervisors manually reviewing hours of dashcam footage, which is 
time-consuming, expensive, and prone to human error. Driving schools, 
insurance companies, and fleet management companies all face this 
same problem — there is no fast, automated way to evaluate whether 
a driver is obeying traffic rules from dashcam footage.

---

## 💡 Approach

To solve this problem, I built an end-to-end AI-powered system 
that automatically analyses dashcam footage and evaluates driver 
behaviour in real time.

### 1. 🗂️ Data Collection & Annotation
- Collected a custom dataset of traffic sign images
- Annotated images with bounding boxes for each sign class:
  - 🔴 Red Light
  - 🟢 Green Light
  - 🛑 Stop Sign
  - 🔢 Speed Limits: 20, 30, 40, 50, 60, 70, 80, 90, 100

### 2. 🧠 Model Training
- Trained a custom **YOLOv8** model on the annotated dataset
- Fine-tuned the model to achieve high accuracy on real road footage
- Exported the best performing weights (`best.pt`) for deployment

### 3. 🎥 Video Processing
- Built a video processing pipeline using **OpenCV**
- The pipeline reads dashcam footage frame by frame
- Each frame is passed through the YOLO model for sign detection
- **Optical Flow** is used to estimate vehicle motion at the moment 
  each sign is detected

### 4. 🔊 Real-Time Voice Alerts
- When a new traffic sign is detected the system triggers a 
  **voice alert** using pyttsx3
- The alert tells the driver what sign was detected
- The same sign is not repeated while it is still visible — 
  only new sign detections trigger a new alert

### 5. 📊 Driver Behaviour Scoring
- The system analyses what the driver did at each sign:
  - Was the vehicle moving or stopped at a red light?
  - Did the driver stop at a stop sign?
  - Was the driver speeding in a speed limit zone?
- A **Safety Score out of 100** is calculated based on violations
- A final verdict is given: Good Driver, Needs Improvement, 
  or Dangerous Driving

### 6. 🖥️ Web Application
- The entire system is wrapped in a clean **Streamlit** web app
- Users can upload any dashcam video or image
- Results are displayed instantly with voice alerts and a 
  final safety score

---

## ✅ Outcomes

- ✔️ Successfully trained a YOLOv8 model to detect 12 traffic 
  sign classes with high accuracy
- ✔️ Built a complete end-to-end pipeline from video input to 
  driver behaviour report
- ✔️ Implemented real-time voice alerts that warn the driver 
  every time a new sign is detected
- ✔️ Developed a scoring system that objectively evaluates 
  driver behaviour based on sign compliance
- ✔️ Deployed the system as an interactive web app using Streamlit
- ✔️ Processing time reduced significantly compared to manual 
  review — a 30 second video is analysed in under 2 minutes

---

## 🌍 Real Life Applications

### 🏫 Driving Schools
Driving instructors can upload a student's dashcam footage and 
instantly get a report showing every traffic sign the student 
encountered and whether they obeyed it. This saves instructors 
hours of manual video review and provides an objective, 
data-driven evaluation of student performance.

### 🚌 Fleet Management
Companies that manage fleets of vehicles such as delivery trucks, 
buses, and taxis can use this system to automatically monitor 
driver behaviour across their entire fleet. Any driver who 
consistently runs red lights or ignores stop signs will be 
flagged immediately without anyone having to watch hours of footage.

### 🛡️ Insurance Companies
Insurance companies can use dashcam footage analysis to objectively 
assess driver risk. A driver with a consistently high safety score 
could be rewarded with lower premiums, while dangerous drivers 
are flagged for intervention.

### 🏙️ Road Safety Research
Urban planners and road safety researchers can use the system to 
analyse large volumes of road footage and identify locations where 
drivers frequently disobey traffic signs — helping prioritise 
where road safety improvements are needed most.

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| YOLOv8 | Custom traffic sign detection |
| OpenCV | Video processing and optical flow |
| pyttsx3 | Offline text to speech voice alerts |
| Streamlit | Interactive web application |
| Python | Core programming language |

---

## ⚙️ Installation & Usage

**1. Clone the repository**
```bash
git clone https://github.com/keshav839/driver-behaviour-analyser.git
cd driver-behaviour-analyser
```

**2. Install dependencies**
```bash
pip install -r Requirements.txt
```

**3. Run the app**
```bash
streamlit run App.py
```

**4. Upload a dashcam video and let the system analyse it!**

---
This the outcome of the system:

<img width="1918" height="904" alt="image" src="https://github.com/user-attachments/assets/6fbda844-81a5-4b02-8c9d-5154ce54b23f" />
<img width="1918" height="898" alt="image" src="https://github.com/user-attachments/assets/02482ce3-a875-4176-86fa-3aa19560363f" />

---
## ⚠️ Limitations & Future Work

- Motion analysis uses optical flow which estimates relative movement 
  and does not measure actual vehicle speed in km/h — GPS integration 
  would make this more precise
- The model performs best on signs similar to the training data — 
  signs from different countries or in poor weather conditions 
  may reduce accuracy
- Voice alerts currently work offline only — future versions will 
  support web-based audio alerts

---

## 👤 Author

**Keshav**  
Computer Vision & Machine Learning 
[GitHub](https://github.com/keshav839)
