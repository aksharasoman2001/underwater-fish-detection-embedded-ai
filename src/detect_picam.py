"""
detect_picam.py — Real-Time Fish Detection on Raspberry Pi 5 + PiCam v3
DARCY: Underwater Fish Detection on Embedded AI
Author: Akshara Soman | Lab-STICC, ENIB Brest | 2025

Usage:
    python src/detect_picam.py
"""

import cv2
import numpy as np
import subprocess
from ultralytics import YOLO

# Load trained YOLOv11 model
model = YOLO("/home/darcy/best.onnx")  # Replace with your correct path


# ─────────────────────────────────────────────
# Supporting Functions
# ─────────────────────────────────────────────

def crop_exact_screen_image(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 10000:
        return None
    epsilon = 0.02 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)
    if len(approx) != 4:
        return None
    pts = approx.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxWidth = int(max(widthA, widthB))
    maxHeight = int(max(heightA, heightB))
    dst = np.array([[0, 0], [maxWidth - 1, 0],
                    [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
                   dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(frame, M, (maxWidth, maxHeight))
    return warped


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    shape = img.shape[:2]
    ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(shape[1] * ratio), int(shape[0] * ratio))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)
    return img, ratio, dw, dh


def preprocess(img):
    return letterbox(img)


# ─────────────────────────────────────────────
# Start PiCam Stream
# ─────────────────────────────────────────────

command = [
    "libcamera-vid", "-t", "0", "--width", "640", "--height", "480",
    "--framerate", "30", "--codec", "yuv420", "-o", "-", "--nopreview"
]
camera = subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=10**8)

while True:
    raw_bytes = camera.stdout.read(640 * 480 * 3 // 2)
    if not raw_bytes:
        break

    yuv = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((int(480 * 1.5), 640))
    bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)

    warped = crop_exact_screen_image(bgr)
    if warped is not None:
        padded_img, ratio, dw, dh = preprocess(warped)

        # YOLO inference
        results = model(padded_img, imgsz=640)[0]

        # Draw detections
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf)
            cls = int(box.cls)
            x1 = int((x1 - dw) / ratio)
            y1 = int((y1 - dh) / ratio)
            x2 = int((x2 - dw) / ratio)
            y2 = int((y2 - dh) / ratio)
            cv2.rectangle(warped, (x1, y1), (x2, y2), (255, 0, 0), 2)
            label = f"{model.names[cls]} {conf:.2f}"
            cv2.putText(warped, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        cv2.imshow("Detection", warped)

    cv2.imshow("Camera Feed", bgr)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.terminate()
cv2.destroyAllWindows()
