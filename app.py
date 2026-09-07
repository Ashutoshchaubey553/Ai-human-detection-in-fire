# app.py (updated to expose per-frame metadata)
import os
import time
import uuid
import threading
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory, Response
from werkzeug.utils import secure_filename

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io
import imutils

# -------------------------
# Config
# -------------------------
BASE_DIR = Path(".")
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "static" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FIRE_MODEL_PATH = MODELS_DIR / "fire_detection_model.h5"
YOLO_CFG = MODELS_DIR / "yolov3.cfg"
YOLO_WEIGHTS = MODELS_DIR / "yolov3.weights"
YOLO_NAMES = MODELS_DIR / "coco.names"

IMG_SIZE = (224, 224)
INVERT_BINARY = False

VIDEO_WORKER_FPS = 20
YOLO_CONF_THRESHOLD = 0.35
YOLO_NMS_THRESHOLD = 0.4
YOLO_INPUT_SIZE = (416, 416)

# -------------------------
# Flask app
# -------------------------
app = Flask(__name__, template_folder="templates", static_folder="static")

# -------------------------
# Load models (fail early if missing)
# -------------------------
if not FIRE_MODEL_PATH.exists():
    raise FileNotFoundError(f"Fire model not found: {FIRE_MODEL_PATH}")
fire_model = load_model(str(FIRE_MODEL_PATH))

if not (YOLO_CFG.exists() and YOLO_WEIGHTS.exists() and YOLO_NAMES.exists()):
    raise FileNotFoundError("YOLO cfg/weights/names files not found in models/")

with open(YOLO_NAMES, "r") as f:
    YOLO_CLASSES = [l.strip() for l in f.readlines() if l.strip()]

yolo_net = cv2.dnn.readNet(str(YOLO_WEIGHTS), str(YOLO_CFG))
PERSON_CLASS_ID = YOLO_CLASSES.index("person")

# -------------------------
# Job store (in-memory)
# Each job: {status, input_path, output_path, progress, latest_frame (bytes), latest_meta, ...}
# -------------------------
_jobs = {}
_jobs_lock = threading.Lock()

# -------------------------
# Helpers & detectors
# -------------------------
def preprocess_fire_frame(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(rgb, IMG_SIZE)
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

def detect_fire(frame_bgr, threshold=0.5):
    x = preprocess_fire_frame(frame_bgr)
    preds = fire_model.predict(x)
    preds = np.array(preds)
    if preds.ndim == 2 and preds.shape[1] == 1:
        model_prob = float(preds[0,0])
        p_fire = model_prob if INVERT_BINARY else (1.0 - model_prob)
    elif preds.ndim == 2 and preds.shape[1] >= 2:
        p_fire = float(preds[0][0])
    else:
        p_fire = 0.0
    conf = round(float(p_fire * 100.0), 2)
    return (p_fire >= threshold), conf

def get_yolo_output_layers_safe(net):
    layer_names = net.getLayerNames()
    out_layers = net.getUnconnectedOutLayers()
    out_layers = np.array(out_layers).flatten()
    return [layer_names[i - 1] for i in out_layers]

def detect_humans_yolo(frame_bgr, conf_thresh=YOLO_CONF_THRESHOLD, nms_thresh=YOLO_NMS_THRESHOLD):
    H, W = frame_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(frame_bgr, 1/255.0, YOLO_INPUT_SIZE, swapRB=True, crop=False)
    yolo_net.setInput(blob)
    outs = yolo_net.forward(get_yolo_output_layers_safe(yolo_net))

    candidates = []
    boxes_xywh = []
    scores = []
    for out in outs:
        for detection in out:
            if len(detection) < 6:
                continue
            obj_conf = float(detection[4])
            class_scores = detection[5:]
            class_id = int(np.argmax(class_scores))
            class_conf = float(class_scores[class_id])
            final_conf = obj_conf * class_conf
            if class_id == PERSON_CLASS_ID and final_conf >= 0.01:
                cx, cy, bw, bh = float(detection[0]), float(detection[1]), float(detection[2]), float(detection[3])
                x1 = int((cx - bw/2) * W)
                y1 = int((cy - bh/2) * H)
                x2 = int((cx + bw/2) * W)
                y2 = int((cy + bh/2) * H)
                x1, y1 = max(0,x1), max(0,y1)
                x2, y2 = min(W,x2), min(H,y2)
                candidates.append((x1,y1,x2,y2,final_conf))
                boxes_xywh.append([x1,y1,x2-x1,y2-y1])
                scores.append(float(final_conf))

    kept = []
    if len(boxes_xywh) > 0:
        try:
            idxs = cv2.dnn.NMSBoxes(boxes_xywh, scores, conf_thresh, nms_thresh)
        except Exception:
            idxs = []
        if len(idxs) > 0:
            idxs_flat = [int(i[0]) if isinstance(i,(list,tuple,np.ndarray)) else int(i) for i in idxs]
            for i in idxs_flat:
                kept.append(candidates[i])
        else:
            kept = [c for c in candidates if c[4] >= conf_thresh]
    return kept

# -------------------------
# Worker: processes video, updates job['latest_frame'] and job['latest_meta']
# -------------------------
def _process_video_job(job_id):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            return
        job["status"] = "running"
        job["started_at"] = time.time()

    input_path = job["input_path"]
    output_path = job["output_path"]

    try:
        cap = cv2.VideoCapture(str(input_path))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = None

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        processed = 0

        while True:
            # cancellation check
            with _jobs_lock:
                cur = _jobs.get(job_id)
                if cur is None:
                    break
                if cur.get("status") == "canceled":
                    cur["message"] = "Canceled by user"
                    break

            ret, frame = cap.read()
            if not ret:
                break

            frame = imutils.resize(frame, width=min(960, frame.shape[1]))
            fire, conf = detect_fire(frame)
            humans = detect_humans_yolo(frame) if fire else []

            # annotate
            label = f"FIRE {conf:.2f}%" if fire else f"NO FIRE {conf:.2f}%"
            color = (0,0,255) if fire else (0,255,0)
            cv2.putText(frame, label, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            for (x1,y1,x2,y2,score) in humans:
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(frame, f"person {score:.2f}", (x1, max(12,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

            # write final file when done
            if writer is None:
                h,w = frame.shape[:2]
                writer = cv2.VideoWriter(str(output_path), fourcc, VIDEO_WORKER_FPS, (w,h))

            writer.write(frame)
            processed += 1

            # update latest_frame for MJPEG streaming
            _, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            jpeg_bytes = jpeg.tobytes()
            # prepare meta
            meta = {"fire": bool(fire), "fire_confidence": float(conf), "humans": int(len(humans))}
            with _jobs_lock:
                if job_id in _jobs:
                    _jobs[job_id]["latest_frame"] = jpeg_bytes
                    _jobs[job_id]["latest_meta"] = meta

            # update progress
            if frame_count > 0 and processed % 5 == 0:
                with _jobs_lock:
                    _jobs[job_id]["progress"] = min(99, int(processed / frame_count * 100))

        cap.release()
        if writer:
            writer.release()

        with _jobs_lock:
            if _jobs[job_id].get("status") == "canceled":
                _jobs[job_id]["finished_at"] = time.time()
                _jobs[job_id]["progress"] = 0
                _jobs[job_id]["message"] = "Canceled by user"
            else:
                _jobs[job_id]["status"] = "done"
                _jobs[job_id]["finished_at"] = time.time()
                _jobs[job_id]["progress"] = 100
                _jobs[job_id]["message"] = "Finished"
    except Exception as e:
        with _jobs_lock:
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["message"] = str(e)

# -------------------------
# Routes
# -------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict_image", methods=["POST"])
def predict_image():
    if "image" not in request.files:
        return jsonify({"error":"no image"}), 400
    f = request.files["image"]
    filename = secure_filename(f.filename) or f"img_{int(time.time())}.jpg"
    in_path = OUTPUT_DIR / f"upload_{uuid.uuid4().hex}_{filename}"
    f.save(str(in_path))

    pil = Image.open(in_path).convert("RGB")
    frame = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    fire, conf = detect_fire(frame)
    humans = detect_humans_yolo(frame) if fire else []

    for (x1,y1,x2,y2,score) in humans:
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, f"person {score:.2f}", (x1, max(12,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0),1)
    label = f"FIRE {conf:.2f}%" if fire else f"NO FIRE {conf:.2f}%"
    col = (0,0,255) if fire else (0,255,0)
    cv2.putText(frame, label, (10,30), cv2.FONT_HERSHEY_SIMPLEX,1, col,2)

    out_name = f"img_out_{uuid.uuid4().hex}.jpg"
    out_path = OUTPUT_DIR / out_name
    cv2.imwrite(str(out_path), frame)

    return jsonify({
        "fire": bool(fire),
        "fire_confidence": float(conf),
        "human_count": len(humans),
        "output_image": f"static/output/{out_name}"
    })

@app.route("/upload_video", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return jsonify({"error":"no video"}), 400
    f = request.files["video"]
    filename = secure_filename(f.filename) or f"video_{int(time.time())}.mp4"
    in_name = f"upload_{uuid.uuid4().hex}_{filename}"
    in_path = OUTPUT_DIR / in_name
    f.save(str(in_path))

    out_name = f"out_{uuid.uuid4().hex}.mp4"
    out_path = OUTPUT_DIR / out_name

    job_id = uuid.uuid4().hex
    job = {
        "status": "queued",
        "input_path": str(in_path),
        "output_path": str(out_path),
        "progress": 0,
        "latest_frame": None,
        "latest_meta": None,
        "started_at": None,
        "finished_at": None,
        "message": ""
    }
    with _jobs_lock:
        _jobs[job_id] = job

    t = threading.Thread(target=_process_video_job, args=(job_id,), daemon=True)
    t.start()

    return jsonify({"job_id": job_id})

@app.route("/job_status/<job_id>", methods=["GET"])
def job_status(job_id):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            return jsonify({"error":"job not found"}), 404
        return jsonify({
            "status": job["status"],
            "progress": job.get("progress",0),
            "output_path": job.get("output_path") if job.get("status")=="done" else None,
            "message": job.get("message",""),
            "latest_meta": job.get("latest_meta")
        })

@app.route("/cancel_job/<job_id>", methods=["POST"])
def cancel_job(job_id):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            return jsonify({"error":"job not found"}), 404
        if job.get("status") in ("done","error","canceled"):
            return jsonify({"status": job.get("status"), "message":"Cannot cancel (already finished)."}), 400
        job["status"] = "canceled"
        job["message"] = "Cancellation requested"
    return jsonify({"status":"canceled", "message":"Cancellation requested"})

# MJPEG stream endpoint for live preview
@app.route("/stream/<job_id>")
def stream_job(job_id):
    with _jobs_lock:
        if job_id not in _jobs:
            return "Job not found", 404

    def generate():
        while True:
            with _jobs_lock:
                job = _jobs.get(job_id)
                if job is None:
                    break
                frame_bytes = job.get("latest_frame")
                status = job.get("status")
            if frame_bytes:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                time.sleep(0.1)
            if status in ("done","error","canceled"):
                break
            time.sleep(0.03)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/output/<path:fname>")
def serve_output(fname):
    return send_from_directory(str(OUTPUT_DIR), fname)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV", "development") == "development"
    app.run(host="0.0.0.0", port=port, debug=debug)
