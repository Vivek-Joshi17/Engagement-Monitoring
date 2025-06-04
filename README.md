# 🎯 Engagement Monitoring System

## 🔍 Overview
The Engagement Monitoring System is a real-time application designed to analyze user engagement during virtual meetings. It detects drowsiness 😴 and spoofing attempts 🕵️‍♂️ (fake or unauthorized users) using advanced computer vision and deep learning techniques, ensuring meeting authenticity and participant attentiveness.

## ✨ Features
- **Drowsiness Detection:** Uses Eye Aspect Ratio (EAR) based facial landmark analysis to detect signs of participant drowsiness in real-time.
- **Spoof Detection:** Employs a YOLOv8n-based deep learning model to identify spoofed faces or fake participants in the meeting.
- **Real-time Processing:** Capable of processing live video streams 📹 and monitoring multiple participants simultaneously.

## 🛠 Technologies Used
- Python 🐍
- OpenCV 📷
- YOLOv8 (Ultralytics) 🎯
- Facial Landmark Detection 🧑‍💻

## ⚙️ How It Works
1. The system captures video frames from each participant's stream.
2. It analyzes facial landmarks to compute EAR and detect drowsiness.
3. Simultaneously, the YOLOv8 model validates participant authenticity by detecting spoofing attempts.
4. Alerts 🚨 are raised when drowsiness or spoofing is detected.

## 🚀 Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the application:
    ```bash
    python app.py
    ```

## 🎬 Usage
- Start a virtual meeting with participants' video feeds.
- The system will monitor each participant in real-time.
- Alerts will notify if drowsiness or spoofing is detected.

## 🔮 Future Improvements
- Support for audio-based engagement analysis 🎙️.
- Integration with popular meeting platforms like Zoom or Microsoft Teams 🤝.
- Enhanced spoof detection with multi-modal biometric checks 🔐.

## 📫 Contact
For questions or contributions, please reach out to [your-email@example.com].

---

*This project was developed by Vivek Joshi.*

