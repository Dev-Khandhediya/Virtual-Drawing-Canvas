# Virtual-Drawing-Canvas
# 🎨 Virtual Drawing Canvas using Computer Vision

A real-time **gesture-controlled virtual drawing application** developed using **Python and OpenCV**.  
This project enables users to draw on a digital canvas **without touching the screen**, using only hand movement captured through a webcam.

The system uses **classical computer vision techniques** such as motion detection, contour analysis, convex hulls, and background subtraction to track
hand movement and convert it into smooth drawing strokes in real time.

---

## 🚀 Features
- 🖐️ Real-time hand tracking using webcam
- 🎨 Multiple color palette selection (Red, Green, Blue, Yellow)
- 🧽 Eraser functionality
- 🔄 Gesture-based drawing mode ON/OFF
- 🧹 Clear canvas option
- ⚡ Smooth drawing with motion threshold filtering
- 🧠 Noise reduction using morphological operations
- 💻 Real-time interactive UI overlay

---

## 🛠️ Tech Stack
### Programming Language
- Python

### Libraries & Tools
- OpenCV (cv2)
- NumPy
- Time (Python Standard Library)

### Concepts Used
- Background Subtraction (MOG2)
- Contour Detection
- Convex Hull
- Morphological Operations
- Motion Tracking
- Real-Time Video Processing

---

### Software Requirements
- Python 3.8 or above

---


## 📦 Installation & Setup

### 1️⃣ Clone the Repository
Open terminal / command prompt and run:
```bash
  -> git clone https://github.com/your-username/Virtual-Drawing-Canvas.git
  -> cd Virtual-Drawing-Canvas

2️⃣ Install Required Libraries

    -> pip install opencv-python numpy

3️⃣ Run the Application

    -> python air_drawing.py
---

🎮 How to Use the Application

- Show your hand in front of the webcam
- Use hand movement to control the cursor
- Select colors from the top palette
- Toggle drawing mode ON/OFF
- Use ERASER to remove strokes
- Click CLEAR to reset the canvas
- Press Q to exit the application


👨‍💻 Author
Dev Khandhediya

LinkedIn: https://www.linkedin.com/in/dev-khandhediya-62a601344
