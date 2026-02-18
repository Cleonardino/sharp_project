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
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Finger Detection</title>
            <style>
                * {
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }
                
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    padding: 20px;
                }
                
                .container {
                    background: white;
                    border-radius: 20px;
                    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
                    padding: 40px;
                    max-width: 900px;
                    width: 100%;
                    margin-top: 20px;
                }
                
                h1 {
                    color: #333;
                    text-align: center;
                    margin-bottom: 30px;
                    font-size: 2.5em;
                    font-weight: 600;
                }
                
                .video-container {
                    position: relative;
                    width: 100%;
                    border-radius: 15px;
                    overflow: hidden;
                    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
                    margin-bottom: 30px;
                }
                
                .video-container img {
                    width: 100%;
                    height: auto;
                    display: block;
                }
                
                .counter-box {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    border-radius: 15px;
                    padding: 30px;
                    text-align: center;
                    box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4);
                }
                
                .counter-label {
                    color: rgba(255, 255, 255, 0.9);
                    font-size: 1.2em;
                    margin-bottom: 10px;
                    text-transform: uppercase;
                    letter-spacing: 2px;
                    font-weight: 500;
                }
                
                .counter-value {
                    color: white;
                    font-size: 4em;
                    font-weight: 700;
                    text-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
                    transition: transform 0.3s ease;
                }
                
                .counter-value.pulse {
                    animation: pulse 0.3s ease;
                }
                
                @keyframes pulse {
                    0%, 100% {
                        transform: scale(1);
                    }
                    50% {
                        transform: scale(1.1);
                    }
                }
                
                .status-indicator {
                    display: inline-block;
                    width: 12px;
                    height: 12px;
                    background: #4ade80;
                    border-radius: 50%;
                    animation: blink 2s infinite;
                    margin-left: 10px;
                }
                
                @keyframes blink {
                    0%, 100% { opacity: 1; }
                    50% { opacity: 0.3; }
                }
                
                @media (max-width: 768px) {
                    .container {
                        padding: 20px;
                    }
                    
                    h1 {
                        font-size: 1.8em;
                    }
                    
                    .counter-value {
                        font-size: 3em;
                    }
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>
                    Finger Detection Dashboard
                    <span class="status-indicator"></span>
                </h1>
                
                <div class="video-container">
                    <img src="/video" alt="Video Feed">
                </div>
                
                <div class="counter-box">
                    <div class="counter-label">Fingers Detected</div>
                    <div class="counter-value" id="count">0</div>
                </div>
            </div>

            <script>
                let lastCount = 0;
                
                async function updateCount() {
                    const res = await fetch('/count');
                    const data = await res.json();
                    const countElement = document.getElementById('count');
                    
                    if (data.fingers !== lastCount) {
                        countElement.classList.add('pulse');
                        setTimeout(() => countElement.classList.remove('pulse'), 300);
                        lastCount = data.fingers;
                    }
                    
                    countElement.innerText = data.fingers;
                }
                
                setInterval(updateCount, 200);
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