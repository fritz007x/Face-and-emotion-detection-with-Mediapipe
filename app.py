import sys
print("Here")
print(sys.executable)
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import mediapipe as mp
from sklearn.metrics.pairwise import cosine_similarity
from fer import FER
import time
import io

# --- Face Detection and Landmarks ---
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
face_detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True, max_num_faces=5, min_detection_confidence=0.6, min_tracking_confidence=0.6)

# Emotion Detector
emotion_detector = FER(mtcnn=True)

st.set_page_config(page_title="Face & Emotion Detection App", layout="wide")
st.title("🧑‍🤝‍🧑 Face & Emotion Detection App")

# --- UX improvement: radio button for image/video selection ---
mode = st.radio("Select input type", ["Images", "Video"])

images = []
video_file = None

if mode == "Images":
    # Sidebar controls
    st.sidebar.header("Detection Settings")
    min_conf = st.sidebar.slider("Min Detection Confidence", 0.0, 1.0, 0.5, 0.01)
    show_landmarks = st.sidebar.toggle("Show Face Landmarks", value=True)
    show_emotions = st.sidebar.toggle("Detect Emotions", value=True)

    uploaded_files = st.file_uploader("Upload Image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    use_webcam = st.checkbox("Use Webcam Feed")
    if uploaded_files:
        for f in uploaded_files:
            img = Image.open(f).convert("RGB")
            images.append(np.array(img))
    if use_webcam:
        frame = st.camera_input("Take a photo")
        if frame is not None:
            img = Image.open(frame).convert("RGB")
            images.append(np.array(img))
elif mode == "Video":
    # Sidebar controls
    st.sidebar.header("Detection Settings")
    min_conf = st.sidebar.slider("Min Detection Confidence", 0.0, 1.0, 0.5, 0.01)
    show_landmarks = st.sidebar.toggle("Show Face Landmarks", value=True)
    show_emotions = st.sidebar.toggle("Detect Emotions", value=True)

    video_file = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])

# Video processing
video_frames = []
video_annotated = []
video_fps = 0
video_out_path = None
if mode == "Video" and video_file is not None:
    import tempfile, shutil, os
    import moviepy.editor as mpy
    temp_dir = tempfile.mkdtemp()
    temp_video_path = os.path.join(temp_dir, video_file.name)
    video_file.seek(0)
    with open(temp_video_path, 'wb') as out_file:
        shutil.copyfileobj(video_file, out_file)
    video = mpy.VideoFileClip(temp_video_path)
    video_fps = video.fps
    for frame in video.iter_frames():
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mesh_result = face_mesh.process(img)
        annotated_img = img.copy()
        if mesh_result.multi_face_landmarks and show_landmarks:
            for face_landmarks in mesh_result.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=annotated_img,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
                mp_drawing.draw_landmarks(
                    image=annotated_img,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                )
        video_annotated.append(cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))
    video_out_path = os.path.join(temp_dir, "annotated_video.mp4")
    mpy.ImageSequenceClip(video_annotated, fps=video_fps).write_videofile(video_out_path, codec="libx264", audio=False)

if mode == "Images" and not images:
    st.info("Upload image(s) or use webcam to get started.")
    st.stop()
if mode == "Video" and video_file is None:
    st.info("Upload a video to get started.")
    st.stop()

# --- Processing indicator ---
results = []
elapsed = 0
try:
    with st.spinner('Processing images, please wait...'):
        start_time = time.time()
        if mode == "Images":
            for idx, img in enumerate(images):
                image_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                detection_result = face_detector.process(image_rgb)
                mesh_result = face_mesh.process(image_rgb)
                faces = []
                landmarks_all = []
                emotions = []
                # Draw bounding boxes and landmarks
                annotated_img = img.copy()
                if detection_result.detections:
                    for det in detection_result.detections:
                        bboxC = det.location_data.relative_bounding_box
                        ih, iw, _ = img.shape
                        x1 = int(bboxC.xmin * iw)
                        y1 = int(bboxC.ymin * ih)
                        w = int(bboxC.width * iw)
                        h = int(bboxC.height * ih)
                        cv2.rectangle(annotated_img, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
                        faces.append(img[y1:y1+h, x1:x1+w])
                        # Draw confidence
                        conf = det.score[0]
                        cv2.putText(annotated_img, f"{conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                if mesh_result.multi_face_landmarks and show_landmarks:
                    for face_landmarks in mesh_result.multi_face_landmarks:
                        mp_drawing.draw_landmarks(
                            image=annotated_img,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                        )
                        mp_drawing.draw_landmarks(
                            image=annotated_img,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                        )
                        landmarks_all.append(face_landmarks)
                if show_emotions and faces:
                    for face in faces:
                        try:
                            emotion, score = emotion_detector.top_emotion(face)
                            emotions.append((emotion, score))
                        except Exception:
                            emotions.append((None, 0))
                results.append({
                    "original": img,
                    "annotated": annotated_img,
                    "faces": faces,
                    "landmarks": landmarks_all,
                    "emotions": emotions,
                    "confidences": [det.score[0] for det in detection_result.detections] if detection_result.detections else []
                })
        elapsed = time.time() - start_time
except Exception as e:
    st.error(f"An error occurred during processing: {e}")

# Display results
if len(results) > 0:
    cols = st.columns(len(results))
    for idx, res in enumerate(results):
        with cols[idx]:
            st.image(res["original"], caption=f"Original Image {idx+1}", use_container_width=True)
            st.image(res["annotated"], caption=f"Detected Faces & Landmarks {idx+1}", use_container_width=True)
            # Show confidence for each detected face
            if res["confidences"]:
                for i, conf in enumerate(res["confidences"]):
                    st.write(f"Face {i+1} Confidence: {conf:.2f}")
            if show_emotions and res["emotions"]:
                st.write("Emotions:")
                for i, (emo, score) in enumerate(res["emotions"]):
                    if score is None:
                        score_display = "N/A"
                    else:
                        score_display = f"{score:.2f}"
                    st.write(f"Face {i+1}: {emo if emo else 'Unknown'} ({score_display})")
            buf = io.BytesIO()
            img = Image.fromarray(res["annotated"])
            img.save(buf, format="PNG")
            buf.seek(0)
            st.download_button(
                label=f"Download Annotated Image {idx+1}",
                data=buf,
                file_name=f"annotated_{idx+1}.png",
                mime="image/png"
            )

# Show annotated video if processed
if video_out_path:
    st.subheader("Annotated Video with FaceMesh")
    st.video(video_out_path)
    with open(video_out_path, "rb") as f:
        st.download_button(
            label="Download Annotated Video",
            data=f,
            file_name="annotated_video.mp4",
            mime="video/mp4"
        )

st.write(f"Processing Time: {elapsed:.2f} seconds")
