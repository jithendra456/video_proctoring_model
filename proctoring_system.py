import cv2
import torch
import numpy as np
import mediapipe as mp
from scipy.spatial import distance as dist
from ultralytics import YOLO
import datetime
import sounddevice as sd
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn
import threading

app = FastAPI()

object_model = YOLO("yolov8n.pt")

drawing_utils = mp.solutions.drawing_utils
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

log_file = open("proctoring_log.txt", "w")

proctoring_active = False

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def lip_distance(landmarks):
    top_lip = landmarks[13]
    bottom_lip = landmarks[14]
    return dist.euclidean(top_lip, bottom_lip)

def get_head_pose(landmarks):
    image_points = np.array([
        landmarks[1], landmarks[152], landmarks[33], landmarks[263], landmarks[61], landmarks[291]
    ], dtype="double")

    model_points = np.array([
        (0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)
    ])

    size = (640, 480)
    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")
    dist_coeffs = np.zeros((4, 1))

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs)

    return rotation_vector, translation_vector

def detect_audio_noise(duration=1, threshold=0.02):
    recording = sd.rec(int(duration * 44100), samplerate=44100, channels=1, dtype='float32')
    sd.wait()
    volume_norm = np.linalg.norm(recording) / len(recording)
    return volume_norm > threshold

def run_proctoring():
    global proctoring_active
    proctoring_active = True

    cap = cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS)

    while proctoring_active:
        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        video_seconds = frame_number / fps
        video_time_formatted = str(datetime.timedelta(seconds=int(video_seconds)))

        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        current_time = datetime.datetime.now()
        timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")

        log_state = {
            "timestamp": timestamp,
            "video_time": video_time_formatted,
            "restricted_objects": [],
            "multiple_persons": 0,
            "lip_movement": False,
            "suspicious_pose": False,
            "audio_noise": detect_audio_noise()
        }

        results = object_model(frame)[0]
        persons_detected = 0
        for r in results.boxes.data.tolist():
            x1, y1, x2, y2, score, cls = map(int, r[:6])
            label = object_model.names[cls]
            if label.lower() == 'person':
                persons_detected += 1
            elif label.lower() in ['cell phone', 'book', 'laptop']:
                log_state["restricted_objects"].append(label)

        if persons_detected > 1:
            log_state["multiple_persons"] = persons_detected

        results = face_mesh.process(frame_rgb)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = [(int(l.x * frame.shape[1]), int(l.y * frame.shape[0]))
                             for l in face_landmarks.landmark]

                lip_dist = lip_distance(landmarks)
                if lip_dist > 20:
                    log_state["lip_movement"] = True

                try:
                    rotation_vector, _ = get_head_pose(landmarks)
                    rot = rotation_vector.flatten()
                    if abs(rot[0]) > 0.5 or abs(rot[1]) > 0.5 or abs(rot[2]) > 0.5:
                        log_state["suspicious_pose"] = True
                except:
                    pass

        if (log_state["restricted_objects"] or log_state["multiple_persons"] > 1
            or log_state["lip_movement"] or log_state["suspicious_pose"] or log_state["audio_noise"]):

            log_entry = f"[{log_state['timestamp']}]"
            if log_state["multiple_persons"]:
                log_entry += f" Multiple persons detected: {log_state['multiple_persons']};"
            if log_state["restricted_objects"]:
                log_entry += f" Restricted objects: {', '.join(log_state['restricted_objects'])};"
            if log_state["lip_movement"]:
                log_entry += " Lip movement detected;"
            if log_state["suspicious_pose"]:
                log_entry += " Suspicious head pose detected;"
            if log_state["audio_noise"]:
                log_entry += " Noise detected in audio;"
            log_file.write(log_entry.strip() + "\n")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    log_file.close()

@app.post("/start-proctoring")
def start_proctoring():
    thread = threading.Thread(target=run_proctoring)
    thread.start()
    return JSONResponse(content={"message": "Proctoring started."})

@app.post("/stop-proctoring")
def stop_proctoring():
    global proctoring_active
    proctoring_active = False
    return JSONResponse(content={"message": "Proctoring stopped."})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
