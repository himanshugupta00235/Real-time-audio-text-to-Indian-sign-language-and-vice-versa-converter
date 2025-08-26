# 🤟 Real-Time Audio/Text to Indian Sign Language and Vice Versa Converter  

This project is a **real-time bi-directional converter** between **Audio/Text** and **Indian Sign Language (ISL)**. It leverages **Speech Recognition, Deep Learning, and Computer Vision** to help bridge communication between hearing and speech-impaired individuals.  

---

## 🔹 Features
- 🎤 **Audio to ISL**: Converts speech into text using **Whisper** and then maps it into Indian Sign Language gestures.  
- ⌨️ **Text to ISL**: Directly converts user-entered text into ISL gesture sequences.  
- ✋ **ISL to Text**: Uses **MediaPipe** and a deep learning model to recognize hand gestures and convert them into text.  
- 📹 **Real-time Video Feed**: OpenCV-based webcam integration for live ISL gesture recognition.  
- 🌐 **Flask Web App**: User-friendly web interface to interact with the system.  

---

## 🔹 Technologies Used
- **Python 3**  
- **Flask** – Web framework for UI integration  
- **Whisper** – Speech-to-text transcription  
- **OpenCV** – Video feed and image processing  
- **MediaPipe** – Hand landmark detection  
- **TensorFlow / Keras** – ISL gesture recognition model  
- **NumPy / Pandas** – Data handling  
- **WTForms** – For file upload and form handling  

---

## 🔹 Project Structure
```
├── app.py                # Main Flask app for audio/text to ISL
├── Part_2.py             # ISL to Text conversion module
├── isl_detection.py      # Real-time hand gesture detection with MediaPipe + CNN model
├── ISL_classifier.ipynb  # Jupyter notebook for training/evaluating ISL recognition model
├── static/               # Static assets (audio, images, CSS, etc.)
├── templates/            # HTML templates for Flask
└── README.md             # Project documentation
```

---

## 🔹 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Real-time-audio-text-to-Indian-sign-language-and-vice-versa-converter.git
   cd Real-time-audio-text-to-Indian-sign-language-and-vice-versa-converter
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # Mac/Linux
   venv\Scripts\activate      # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🔹 Usage

1. Start the Flask server:
   ```bash
   python app.py
   ```

2. Open the app in your browser at:
   ```
   http://127.0.0.1:5000/
   ```

3. Navigate between features:
   - **Part 1 (Audio/Text → ISL)**: Upload audio or type text to convert into ISL gestures.  
   - **Part 2 (ISL → Text)**: Show ISL gestures via webcam for real-time text output.  

---

## 🔹 Workflow
1. **Audio to Text** → Whisper model transcribes speech into text.  
2. **Text to ISL** → The text is mapped to ISL gestures using a predefined mapping.  
3. **ISL Gesture Detection** → MediaPipe extracts hand landmarks → preprocessed → passed into CNN model.  
4. **Gesture Classification** → TensorFlow/Keras model predicts ISL alphabet/number → converts into text.  

---

## 🔹 Future Improvements
- Add **support for continuous ISL sentence recognition** (not just alphabet-level).  
- Improve gesture recognition accuracy with larger datasets.  
- Deploy as a **web service** or **mobile app** for accessibility.  

---

## 🔹 Screenshots / Demo
(Add your screenshots or demo GIFs here showing audio-to-ISL and ISL-to-text in action)  

---

## 🔹 Author
👤 **Himanshu Gupta**  
📧 Email: [your_email@example.com]  
🌐 GitHub: [himanshugupta00235](https://github.com/himanshugupta00235)  

---

⭐ If you found this project useful, don’t forget to **star this repository**!
