# Face and Emotion Detection with Mediapipe and FER

This is a web app developed for the course *CAI2840C-2253-7384 Introduction to Computer Vision.*
It is a face and emotion detection application that uses:

* Google Mediapipe (for face detection and facial landmarks)
* FER (Facial Expression Recognition) by Justin Shenk (for emotion detection)

The frontend is implemented using Streamlit.

## Features

* Upload or capture photos and videos using a webcam.

* Detect faces in images and videos, with bounding boxes highlighting detected faces.

* Display confidence scores for both face detection and emotion recognition.

* Show facial landmarks and face meshes:

*  The Face Mesh model provides 468 three-dimensional facial landmarks (The more advanced version "FaceLandmarker v2" detects 478 landmarks, including 10 around the irises for better eye tracking.)

* Download the processed images and videos.


## How to Run the App Locally

**Requirements:**
* Python 3.8 or newer
* Install dependencies from requirements.txt

```markdown
```python
# Clone the repository and move to the project folder
cd your_project_folder

# (Optional) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

**App Usage:**

1. Select Images or Video input mode.

2. Upload files or capture a new photo/video with your webcam.

3. Adjust settings (confidence thresholds, show landmarks, detect emotions) in the sidebar.

4. View results: original and annotated images/videos, face detection confidence, detected emotions.

5. Download the processed files.


## Libraries Used
* streamlit

* mediapipe

* opencv-python

* numpy

* Pillow

* scikit-learn

* fer
