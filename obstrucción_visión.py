
# -*- coding: utf-8 -*-
import cv2
import mediapipe as mp
import time
import numpy as np
from pygame import mixer
import os

# Inicializar pygame mixer
mixer.init()

# Cargar sonido de alerta
alert_path = 'alerta2.wav'
if not os.path.exists(alert_path):
    raise FileNotFoundError(f"No se encontró el archivo '{alert_path}'")

alert_sound = mixer.Sound(alert_path)

# Inicializar MediaPipe
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                  min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Constantes
EYE_AR_THRESH = 0.25
EYE_CLOSED_SECONDS = 0.5

# EAR
def eye_aspect_ratio(eye_indices, landmarks, frame_shape):
    try:
        p1 = np.array([landmarks[eye_indices[0]].x * frame_shape[1], landmarks[eye_indices[0]].y * frame_shape[0]])
        p2 = np.array([landmarks[eye_indices[1]].x * frame_shape[1], landmarks[eye_indices[1]].y * frame_shape[0]])
        p3 = np.array([landmarks[eye_indices[2]].x * frame_shape[1], landmarks[eye_indices[2]].y * frame_shape[0]])
        p4 = np.array([landmarks[eye_indices[3]].x * frame_shape[1], landmarks[eye_indices[3]].y * frame_shape[0]])
    except IndexError:
        return None

    vert_dist = np.linalg.norm(p3 - p4)
    horz_dist = np.linalg.norm(p1 - p2)

    if horz_dist == 0:
        return None
    return vert_dist / horz_dist

def is_hand_covering_eyes(eye_points, hand_landmarks, w, h):
    eye_x = np.mean([pt.x for pt in eye_points])
    eye_y = np.mean([pt.y for pt in eye_points])
    eye_pos = np.array([eye_x * w, eye_y * h])

    for lm in hand_landmarks.landmark:
        hand_pos = np.array([lm.x * w, lm.y * h])
        if np.linalg.norm(hand_pos - eye_pos) < 40:
            return True
    return False

def are_dark_glasses_detected(eye_points, frame, threshold=60):
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    points = np.array([[int(p.x * frame.shape[1]), int(p.y * frame.shape[0])] for p in eye_points])
    cv2.fillConvexPoly(mask, points, 255)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eye_region = cv2.bitwise_and(gray, gray, mask=mask)

    mean_brightness = cv2.mean(eye_region, mask=mask)[0]
    return mean_brightness < threshold

# Variables
cap = cv2.VideoCapture(0)
eyes_closed = False
closed_eye_start = None
alarm_triggered = False

# Captura en tiempo real
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_face = face_mesh.process(frame_rgb)
    results_hands = hands.process(frame_rgb)
    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    h, w = frame.shape[:2]

    eyes_detected = False
    ear = 1.0
    vision_blocked = False
    dark_glasses = False

    if results_face.multi_face_landmarks:
        landmarks = results_face.multi_face_landmarks[0].landmark

        left_indices = [33, 133, 159, 145]
        right_indices = [362, 263, 386, 374]

        left_ear = eye_aspect_ratio(left_indices, landmarks, frame.shape)
        right_ear = eye_aspect_ratio(right_indices, landmarks, frame.shape)

        if left_ear is not None and right_ear is not None:
            ear = (left_ear + right_ear) / 2.0
            eyes_detected = True

            for idx in left_indices + right_indices:
                x = int(landmarks[idx].x * w)
                y = int(landmarks[idx].y * h)
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        left_eye_points = [landmarks[i] for i in left_indices]
        right_eye_points = [landmarks[i] for i in right_indices]

        if results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                if is_hand_covering_eyes(left_eye_points, hand_landmarks, w, h) or \
                   is_hand_covering_eyes(right_eye_points, hand_landmarks, w, h):
                    vision_blocked = True

        # Detectar gafas oscuras
        dark_glasses = are_dark_glasses_detected(left_eye_points + right_eye_points, frame)

    # Evaluación de condiciones
    if not eyes_detected or ear < EYE_AR_THRESH or vision_blocked or dark_glasses:
        if not eyes_closed:
            closed_eye_start = time.time()
            eyes_closed = True
        elif time.time() - closed_eye_start >= EYE_CLOSED_SECONDS:
            cv2.putText(frame, "¡ALERTA!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if dark_glasses:
                cv2.putText(frame, "GAFAS OSCURAS DETECTADAS", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if not alarm_triggered:
                alert_sound.play(loops=-1)
                alarm_triggered = True
    else:
        eyes_closed = False
        closed_eye_start = None
        if alarm_triggered:
            alert_sound.stop()
        alarm_triggered = False

    # Mostrar EAR
    cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Mostrar ventana
    cv2.imshow("Obstrucción de la Visión", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
face_mesh.close()
hands.close()
