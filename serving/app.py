import cv2
import atexit
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from ultralytics import YOLO

# -----------------------------
# Initialisation
# -----------------------------
app = FastAPI()

model = YOLO("best.pt")

# Backend caméra Windows (DirectShow)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise RuntimeError("Impossible d'accéder à la webcam")

# Libération propre de la webcam
@atexit.register
def cleanup():
    cap.release()

# Variable globale pour le compteur
current_finger_count = 0

# -----------------------------
# Générateur de frames MJPEG
# -----------------------------
def gen_frames():
    global current_finger_count
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO inference
        results = model(frame, verbose=False)[0]
        finger_count = 0
        for box in results.boxes:
            finger_count += int(model.names[int(box.cls[0])].lower())
        current_finger_count = finger_count  # stocke pour l'API / HTML

        # Dessiner les bounding boxes
        for box in results.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Afficher le compteur sur l'image
        cv2.putText(frame, f"Total fingers: {finger_count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        _, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")


# -----------------------------
# Endpoints FastAPI
# -----------------------------
@app.get("/")
def index():
    return HTMLResponse(
        """
        <!DOCTYPE html>
        <html lang="fr">
        <head>
            <meta charset="UTF-8">
            <title>Finger Detection</title>
        </head>
        <body style="text-align:center; font-family:Arial;">
            <h1>Finger Detection Dashboard</h1>
            <img src="/video" width="800"><br>
            <h2>Fingers detected: <span id="count">0</span></h2>

            <script>
                async function updateCount() {
                    const res = await fetch('/count');
                    const data = await res.json();
                    document.getElementById('count').innerText = data.fingers;
                }
                setInterval(updateCount, 200); // mise à jour toutes les 200ms
            </script>
        </body>
        </html>
        """
    )


@app.get("/video")
def video_feed():
    return StreamingResponse(
        gen_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/count")
def get_count():
    return JSONResponse({"fingers": current_finger_count})