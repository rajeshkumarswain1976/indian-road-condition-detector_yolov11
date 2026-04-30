import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("runs/detect/model2/weights/best.pt")

CLASS_WEIGHTS = {0:1.5, 1:2.0, 2:1.2}

def compute_severity(results, shape):
    h, w = shape[:2]
    total = h*w
    score = 0

    for box in results.boxes:
        cls = int(box.cls[0])
        x1, y1, x2, y2 = box.xyxy[0]
        area = (x2-x1)*(y2-y1)
        score += (area/total)*CLASS_WEIGHTS[cls]

    return score

def classify(score):
    if score < 0.03:
        return "GOOD"
    elif score < 0.1:
        return "MODERATE"
    else:
        return "DANGEROUS"

def heatmap(img, results):
    heat = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        heat[y1:y2, x1:x2] += 1

    heat = cv2.normalize(heat, None, 0, 255, cv2.NORM_MINMAX)
    heat = cv2.applyColorMap(heat.astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.addWeighted(img, 0.6, heat, 0.4, 0)

st.title("Road Damage Detection System")

file = st.file_uploader("Upload Image", type=["jpg","png"])

if file:
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), 1)

    results = model(img)[0]

    score = compute_severity(results, img.shape)
    condition = classify(score)

    st.image(img, caption="Original")
    st.image(results.plot(), caption="Detection")
    st.image(heatmap(img, results), caption="Heatmap")

    st.write(f"Score: {score:.4f}")
    st.write(f"Condition: {condition}")
