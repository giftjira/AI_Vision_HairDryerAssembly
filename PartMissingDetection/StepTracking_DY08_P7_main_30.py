#!/usr/bin/env python3
"""
Real‑time step tracking with TFLite + MQTT (hardened)
- Robust boot even before MQTT sends Model_ID
- Safe concurrent model reloads (lock)
- Correct YOLO output scaling (no hard‑coded 256)
- Rebuilds Config/StateMachine when Model_ID changes
- Graceful CSV parsing from filename (with fallbacks)
- Optional alarm sound (won't crash if missing)
- Red overlay when any class is Skipped
- Defensive guards around camera / window init

Keys during runtime:
  q       Quit
  +/-     Detection threshold ↑/↓
  ,/.     IoU (NMS) threshold ↑/↓
  [/]     Score threshold ↑/↓
"""
import os
import re
import cv2
import ast
import json
import time
import glob
import csv
import numpy as np
import pandas as pd
from typing import Dict
from pathlib import Path
from threading import Thread, Lock
import subprocess
# ──────────────────────────────────────────────────────────────────────────────
# Optional audio (won't crash if library not installed)
# ──────────────────────────────────────────────────────────────────────────────
try:
    import simpleaudio as sa
except Exception:  # pragma: no cover
    sa = None

ALARM_SOUND_PATH = os.getenv("ALARM_WAV", "alarm.wav")

# ──────────────────────────────────────────────────────────────────────────────
# MQTT (optional but enabled by default)
# ──────────────────────────────────────────────────────────────────────────────
import paho.mqtt.client as mqtt

HOST_ADDR = os.getenv("MQTT_HOST", "10.84.171.108")
FIREWALL_PORT = int(os.getenv("MQTT_PORT", 1883))

# ──────────────────────────────────────────────────────────────────────────────
# TFLite interpreter
# ──────────────────────────────────────────────────────────────────────────────
try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
except Exception:
    from tensorflow.lite import Interpreter  # type: ignore
    load_delegate = None

# ──────────────────────────────────────────────────────────────────────────────
# Globals guarded by a lock when hot‑swapped from MQTT
# ──────────────────────────────────────────────────────────────────────────────
interpreter = None
input_details = None
output_details = None
cfg = None
sm = None
objects = []
detection_history: Dict[str, list] = {}
detection_status: Dict[str, str] = {}
interpreter_lock = Lock()

# Alert state (for on-screen missing notification)
alert_end_time = 0.0
alert_lines = []

# Recording configuration
END_STATE_NAME = os.getenv("END_STATE_NAME", "Box Closing")
RECORD_ROOT = Path(os.getenv("NAS_PATH","/mnt/nas/Skipped_Video") or os.getenv("RECORD_DIR", "records"))
RECORD_ROOT.mkdir(parents=True, exist_ok=True)
RECORD_LOG_CSV = Path(os.getenv("RECORD_LOG_CSV", RECORD_ROOT / "skip_log.csv"))
RECORD_CODEC = os.getenv("RECORD_CODEC", "mp4v")
RECORD_EXT = os.getenv("RECORD_EXT", ".mp4")
END_STATE_RESOLVED = None
# Sensible defaults so the app can boot before MQTT arrives
Model_ID = None #os.getenv("MODEL_ID")
std_time = None

# ──────────────────────────────────────────────────────────────────────────────
# Filename → line/site inference (safe fallbacks)
# ──────────────────────────────────────────────────────────────────────────────
FALLBACK_FILENAME = os.getenv("SCRIPT_NAME", "script_LINE_SITE.py")
filename = os.path.basename(globals().get("__file__", FALLBACK_FILENAME))
parts = filename.split("_")
line_id = parts[1] if len(parts) > 1 else os.getenv("LINE_ID", "N/A")
site_id = parts[2] if len(parts) > 2 else os.getenv("SITE_ID", "NA")

# ──────────────────────────────────────────────────────────────────────────────
# Config parsing helpers
# ──────────────────────────────────────────────────────────────────────────────
N_HISTORY = 5
SCALE_FACTOR = float(os.getenv("DISPLAY_SCALE", 1.5))

model_thresholds = {
    "model": {
        "DETECTION_THRESHOLD": 0.85,
        "IOU_THRESHOLD": 0.20,
        "SCORE_THRESHOLD": 0.85,
    }
}

class_colors = {
    "Filler & Individual": (255, 0, 0),
    "Main Body": (0, 255, 0),
    "Nozzle 1": (0, 0, 255),
    "Nozzle 2": (255, 255, 0),
    "Nozzle 3": (255, 0, 0),
    "Web Card & Kit Set": (255, 0, 255),
    "Box Closing": (0, 255, 255),
}


def parse_dict_like(src):
    if isinstance(src, dict):
        return src
    s = str(src).strip()
    try:
        return ast.literal_eval(s)
    except Exception:
        pass
    s_fixed = re.sub(r"([{,]\s*)([A-Za-z0-9_]+)\s*:", r'\1"\2":', s)
    s_fixed = s_fixed.replace("'", '"')
    return json.loads(s_fixed)

def _resolve_end_state(cfg, desired: str) -> str:
    # If desired matches a configured state directly, use it.
    if desired in cfg.states:
        return desired
    # Otherwise, try to interpret it as a CLASS name
    if desired in cfg.name_to_idx:
        idx0 = cfg.name_to_idx[desired]  # 0-based
        st = cfg.class_to_state.get(idx0 + 1)  # mapping uses 1-based keys
        if st:
            return st
    # Fallback: return unchanged (may be wrong, but won’t crash)
    return desired

class Config:
    def __init__(self, csv_path, model_id):
        df = pd.read_csv(Path(csv_path))
        try:
            row = df.loc[df["Model_ID"] == model_id].iloc[0]
        except IndexError:
            raise ValueError(f"Model_ID '{model_id}' not found in {csv_path}")

        cls_map = parse_dict_like(row["class_index"])  # {0: name, ...}
        self.idx_to_name = {int(k) + 1: v for k, v in cls_map.items()}  # 1‑based
        self.name_to_idx = {v: int(k) for k, v in cls_map.items()}      # 0‑based
        self.n_classes = len(self.idx_to_name)

        state_tbl = parse_dict_like(row["state_class"])  # {state: [..]}
        self.states = list(state_tbl.keys())
        self.det_mat = np.array([state_tbl[s] for s in self.states], dtype=int)
        self.state_to_row = {s: i for i, s in enumerate(self.states)}

        home_map = parse_dict_like(row["class_index_list"])  # {state: [idx0,..]}
        self.class_to_state = {}
        for state, lst in home_map.items():
            for idx in lst:
                self.class_to_state[idx + 1] = state  # 1‑based key
        self.home_map = home_map


class StateMachine:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.cur_state = None
        self.cur_row_idx = None
        self.skipped = set()

    def _pick_start_state(self, rt_arr):
        best = None  # (highest_class_idx, row_idx)
        for r, _state in enumerate(self.cfg.states):
            prod = self.cfg.det_mat[r] * rt_arr
            plus2_idx = np.where(prod == 2)[0]
            if plus2_idx.size:
                cand = (plus2_idx.max(), r)
                if best is None or cand > best:
                    best = cand
        if best:
            _, r = best
            self.cur_row_idx = r
            self.cur_state = self.cfg.states[r]

    def _advance(self):
        self.cur_row_idx = (self.cur_row_idx + 1) % len(self.cfg.states)
        self.cur_state = self.cfg.states[self.cur_row_idx]
        self.skipped.clear()

    def _change_to_class_state(self, class_idx_zero_based):
        new_state = self.cfg.class_to_state[class_idx_zero_based + 1]
        self.cur_row_idx = self.cfg.state_to_row[new_state]
        self.cur_state = new_state

    def update(self, rt_arr):
        if self.cur_state is None:
            self._pick_start_state(rt_arr)
            if self.cur_state is None:
                self.cur_row_idx = 0
                self.cur_state = self.cfg.states[0]
            return self.cur_state

        row = self.cfg.det_mat[self.cur_row_idx]
        prod = row * rt_arr

        # Skip if any +3
        skip_idx = np.where(prod == 3)[0]
        if skip_idx.size:
            self.skipped.update(np.where(prod == -2)[0])
            self._change_to_class_state(skip_idx.max())
            return self.cur_state

        # Fallback if any required (row==1) is -1 (not seen)
        fb_idx = np.where((row == 1) & (prod == -1))[0]
        if fb_idx.size:
            self._change_to_class_state(fb_idx.max())
            return self.cur_state

        # Advance if no -2 remaining
        if not np.any(prod == -2):
            self._advance()
        return self.cur_state

    def build_status_dict(self, rt_arr):
        if self.cur_state is None:
            return {name: "???" for name in self.cfg.idx_to_name.values()}
        row = self.cfg.det_mat[self.cur_row_idx]
        prod = (row * rt_arr).ravel()
        status = {}
        for i in range(self.cfg.n_classes):
            name = self.cfg.idx_to_name[i + 1]
            v = int(prod[i])
            if v in (2, 3, 1):
                status[name] = "Found"
            elif i in self.skipped:
                status[name] = "Skipped"
            elif v == -2:
                status[name] = "Waiting"
            else:
                status[name] = "???"
        return status


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def _find_config_path() -> Path:
    env = os.getenv("CONFIG_CSV")
    if env and Path(env).exists():
        return Path(env)
    candidate = Path(f"config_{line_id}_{site_id}_StepTracking.csv")
    if candidate.exists():
        return candidate
    # Last resort: glob for any matching line id
    matches = sorted(glob.glob(f"config_{line_id}_*_StepTracking.csv"))
    if matches:
        return Path(matches[0])
    raise FileNotFoundError(
        "Could not locate config CSV. Set CONFIG_CSV env or follow naming 'config_{line_id}_{site}_StepTracking.csv'."
    )

def load_interpreter_with_fallback(model_id: str) -> Interpreter:
    new_path = Path(f"models/{model_id}/model_{model_id}.tflite")
    default_path = Path(os.getenv("DEFAULT_TFLITE", "model_NA0K-P-fp16.tflite"))
    path = new_path if new_path.exists() else default_path
    if new_path.exists():
        print(f"[INFO] Loading model: {new_path}")
    else:
        print(f"[WARN] {new_path} not found. Falling back to {default_path}")

    num_threads = int(os.getenv("TFLITE_THREADS", max(1, (os.cpu_count() or 4)//2)))
    delegates = []
    if os.getenv("TFLITE_GPU", "0") == "1" and load_delegate is not None:
        try:
            delegates.append(load_delegate("libtensorflowlite_gpu_delegate.so"))
            print("[INFO] Using TFLite GPU delegate")
        except Exception as e:
            print("[WARN] GPU delegate not available:", e)

    return Interpreter(model_path=str(path),
                       experimental_delegates=delegates or None,
                       num_threads=num_threads)

def get_input_output_details(interpreter: Interpreter, model_name: str):
    inp = interpreter.get_input_details()
    out = interpreter.get_output_details()
    print("-" * 70)
    print(f"{model_name} Output Details:", out)
    print("-" * 70)
    print(f"{model_name} Input Details:", inp)
    print("-" * 70)
    return inp, out

def rebuild_objects_dicts(_cfg: Config):
    global objects, detection_history, detection_status
    objects = [
        _cfg.idx_to_name[int(i) + 1]
        for st in _cfg.states
        for i in _cfg.home_map[st]
    ]
    detection_history = {name: [0] * N_HISTORY for name in objects}
    detection_status = {name: "???" for name in objects}

# ──────────────────────────────────────────────────────────────────────────────
# MQTT handlers
# ──────────────────────────────────────────────────────────────────────────────

def on_connect(client, userdata, flags, rc):
    print("MQTT connected with result code:", rc)
    if rc == 0:
        client.subscribe(f"{line_id}/Model_ID")
        print(f"Subscribed to topic: {line_id}/Model_ID")


def on_message(client, userdata, msg):
    global Model_ID, std_time
    topic = msg.topic
    payload = msg.payload.decode()
    print(f"[DEBUG] Received on {topic}: {payload}")
    if topic.endswith("Model_ID"):
        parts = str(payload).split("_")
        Model_ID = parts[0]
        std_time = parts[1] if len(parts) > 1 else None
        print(f"[INFO] New Model_ID: {Model_ID}")
        reload_everything_for_new_model_id()


client = mqtt.Client(protocol=mqtt.MQTTv311)
client.on_connect = on_connect
client.on_message = on_message
try:
    client.connect(HOST_ADDR, FIREWALL_PORT, 60)
    client.loop_start()
except Exception as e:  # pragma: no cover
    print("[WARN] MQTT connect failed:", e)


# ──────────────────────────────────────────────────────────────────────────────
# Model + config hot‑reload
# ──────────────────────────────────────────────────────────────────────────────

def reload_everything_for_new_model_id():
    global interpreter, input_details, output_details, cfg, sm, END_STATE_RESOLVED
    with interpreter_lock:
        cfg_path = _find_config_path()
        cfg = Config(cfg_path, Model_ID)
        sm = StateMachine(cfg)
        rebuild_objects_dicts(cfg)
        interpreter = load_interpreter_with_fallback(Model_ID)
        interpreter.allocate_tensors()
        input_details, output_details = get_input_output_details(interpreter, Model_ID)
        END_STATE_RESOLVED = _resolve_end_state(cfg, END_STATE_NAME)
    print(f"[INFO] Reload complete. END_STATE_NAME='{END_STATE_NAME}' → resolved to state '{END_STATE_RESOLVED}'")

# Initial load so we can start immediately
#reload_everything_for_new_model_id()

def mount_nas():
    try:
        subprocess.run(['sudo','mkdir','-p','/mnt/nas'],check=True)

        subprocess.run(['sudo','mount','-t','cifs'
                   ,'//158.118.87.150/pmfth-nas/CameraCapture-AI-Vision'
                   ,'/mnt/nas','-o'
                   ,'username=09PCB,password=Edpisc#01,uid=1000,gid=1000,rw,vers=3.0'],check=True)
    except subprocess.CalledProcessError as e:
        print()

# ──────────────────────────────────────────────────────────────────────────────
# Video utils
# ──────────────────────────────────────────────────────────────────────────────
class VideoStream:
    def __init__(self, src, width=640, height=480):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()

# ──────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ──────────────────────────────────────────────────────────────────────────────

def preprocess_frame(frame, input_details):
    h_in, w_in = input_details[0]["shape"][1:3]
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h0, w0 = img.shape[:2]
    s = min(w_in / w0, h_in / h0)
    nw, nh = int(round(w0 * s)), int(round(h0 * s))
    img = cv2.resize(img, (nw, nh))
    canvas = np.full((h_in, w_in, 3), 114, dtype=np.uint8)
    dw, dh = (w_in - nw) // 2, (h_in - nh) // 2
    canvas[dh : dh + nh, dw : dw + nw] = img

    if input_details[0]["dtype"] == np.uint8:
        tensor = canvas.astype(np.uint8)
    else:
        tensor = canvas.astype(np.float32) / 255.0
    return np.expand_dims(tensor, 0), (s, dw, dh, h0, w0, w_in, h_in)


def apply_nms_per_class(boxes, scores, classes, iou_threshold, score_threshold):
    final_boxes, final_scores, final_classes = [], [], []
    for cls in set(classes):
        idxs = [i for i, c in enumerate(classes) if c == cls]
        cls_boxes = [boxes[i] for i in idxs]
        cls_scores = [float(scores[i]) for i in idxs]
        xywh = []
        for x_min, y_min, x_max, y_max in cls_boxes:
            xywh.append([int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)])
        idx = cv2.dnn.NMSBoxes(xywh, cls_scores, score_threshold, iou_threshold)
        if len(idx) > 0:
            idx = idx.flatten()
            for i in idx:
                final_boxes.append(cls_boxes[i])
                final_scores.append(cls_scores[i])
                final_classes.append(cls)
    return final_boxes, final_scores, final_classes

def process_yolov5_output(output_data, meta, quant_params, detection_threshold):
    scale, dw, dh, h0, w0, w_in, h_in = meta
    s, zp = quant_params
    if s:
        output_data = s * (output_data - zp)
    if output_data.ndim > 2:
        output = output_data.reshape(-1, output_data.shape[-1])
    elif output_data.ndim == 2:
        output = output_data
    else:
        output = output_data[None, :]

    # Filter by objectness
    obj_conf = output[:, 4]
    keep = obj_conf >= detection_threshold
    if not np.any(keep):
        return [], [], []

    output = output[keep]
    obj_conf = obj_conf[keep]

    # Class score & id (vectorized)
    cls_scores = output[:, 5:]
    cls_id = np.argmax(cls_scores, axis=1)
    cls_conf = cls_scores[np.arange(cls_scores.shape[0]), cls_id]
    score = obj_conf * cls_conf
    keep2 = score >= detection_threshold
    if not np.any(keep2):
        return [], [], []

    output = output[keep2]
    score = score[keep2]
    cls_id = cls_id[keep2]

    # Boxes: (x,y,w,h) → (xmin,ymin,xmax,ymax) in input space, then de-letterbox
    x_c, y_c, w, h = output[:, 0], output[:, 1], output[:, 2], output[:, 3]
    x_min = (x_c - w/2.0) * w_in
    y_min = (y_c - h/2.0) * h_in
    x_max = (x_c + w/2.0) * w_in
    y_max = (y_c + h/2.0) * h_in

    x_min = ((x_min - dw) / scale).clip(0, w0 - 1)
    y_min = ((y_min - dh) / scale).clip(0, h0 - 1)
    x_max = ((x_max - dw) / scale).clip(0, w0 - 1)
    y_max = ((y_max - dh) / scale).clip(0, h0 - 1)

    boxes = np.stack([x_min, y_min, x_max, y_max], axis=1).astype(np.int32)

    # Optional: pre-select top-K before NMS to cut work
    topk = int(os.getenv("NMS_TOPK", "200"))
    if boxes.shape[0] > topk:
        idx = np.argpartition(score, -topk)[-topk:]
        boxes, score, cls_id = boxes[idx], score[idx], cls_id[idx]

    # Return numpy arrays; your NMS fn will handle per-class
    return boxes.tolist(), score.tolist(), cls_id.tolist()

def draw_detection(frame, xmin, ymin, xmax, ymax, class_name, score):
    color = class_colors.get(class_name, (0, 255, 0))
    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
    label = f"{class_name}: {int(score * 100)}%"
    (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    y = max(ymin, th + 10)
    cv2.rectangle(frame, (xmin, y - th - 10), (xmin + tw, y + base - 10), (255, 255, 255), cv2.FILLED)
    cv2.putText(frame, label, (xmin, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)


def play_alarm():  # non‑blocking by caller
    if sa is None:
        print("[ALERT] Skipped item detected (audio lib not available)")
        return
    try:
        wave_obj = sa.WaveObject.from_wave_file(ALARM_SOUND_PATH)
        play_obj = wave_obj.play()
        play_obj.wait_done()
    except Exception as e:  # pragma: no cover
        print("[WARN] Error playing alarm:", e)


def overlay_red_alert(frame, alpha=0.3):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return frame


# --- Skipped/Missing alert helpers ---

def trigger_skip_alert(detection_status, duration=3.0):
    """Arm a 3s screen alert listing items that were SKIPPED or WAITING and play a sound."""
    global alert_end_time, alert_lines
    missing = [name for name, st in detection_status.items() if st in ("Skipped", "Waiting")]
    if not missing:
        return
    alert_lines = missing
    alert_end_time = time.time() + float(duration)
    Thread(target=play_alarm, daemon=True).start()


def render_missing_alert(frame):
    """Render a semi-transparent overlay with the list of missing/skimmed items
    while the alert window is active."""
    if time.time() >= alert_end_time:
        return
    # Dim background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
    # Header
    cv2.putText(frame, "MISSING / SKIPPED / WAITING", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    y = 100
    for line in alert_lines:
        cv2.putText(frame, f"- {line}", (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        y += 35

# --- Recording helpers ---

def _get_dated_log_path(record_path: Path) -> Path:
    date_str = time.strftime("%Y%m%d", time.localtime())
    env_path = os.getenv("RECORD_LOG_CSV")
    if env_path:
        p = Path(env_path)
        if p.suffix.lower() == ".csv":
            return p.with_name(f"{p.stem}_{date_str}{p.suffix}")
        else:
            return p / f"skip_log_{date_str}.csv"
    return Path(record_path).parent / f"skip_log_{date_str}.csv"

def _append_skip_log(time_error_ts: float, record_path: Path):
    try:
        log_path = _get_dated_log_path(record_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not log_path.exists()

        # ISO 24h (stable for parsers)
        ts_iso = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time_error_ts))
        # Legacy 12h (human-friendly)
        try:
            ts_legacy = time.strftime("%-m/%-d/%Y %I:%M:%S %p", time.localtime(time_error_ts))
        except ValueError:
            ts_legacy = time.strftime("%m/%d/%Y %I:%M:%S %p", time.localtime(time_error_ts))

        filename_only = Path(record_path).name  # <-- just the filename

        with open(log_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(["timestamp", "record_filename", "time_error"])
            w.writerow([ts_iso, filename_only, ts_legacy])
            try:
                f.flush()
                os.fsync(f.fileno())
            except Exception:
                pass

        print(f"[LOG] Wrote skip entry: {log_path} -> {ts_iso}, {filename_only}")

    except Exception as e:
        print("[WARN] Failed writing skip log:", e)


# ──────────────────────────────────────────────────────────────────────────────
# UI helpers
# ──────────────────────────────────────────────────────────────────────────────

def draw_status_list(frame, detection_status, current_state, obj_list, cfg):
    h, w, _ = frame.shape
    x, y = 15, 320
    line_h = 22
    active_set = {cfg.idx_to_name[int(i)] for i, st in cfg.class_to_state.items() if st == current_state}

    for obj in obj_list:
        status = detection_status.get(obj, "???")
        text_color, bg_color = (255, 255, 255), (128, 128, 128)
        if status == "Found":
            text_color, bg_color = (0, 0, 0), (0, 255, 0)
        elif status == "Waiting":
            text_color, bg_color = (0, 0, 0), (0, 165, 255)
        elif status == "Skipped":
            text_color, bg_color = (255, 255, 255), (0, 0, 255)

        if obj in active_set:
            font, fs, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        else:
            font, fs, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1

        text = f"> {obj}: {status}"
        (tw, th), base = cv2.getTextSize(text, font, fs, thick)
        cv2.rectangle(frame, (x, y - th - base), (x + tw, y + base), bg_color, cv2.FILLED)
        cv2.putText(frame, text, (x, y), font, fs, text_color, thick)
        y += line_h
    return frame


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    global detection_status, END_STATE_RESOLVED

    #Mount NAS
    mount_nas()
    try:
        RECORD_ROOT.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print("[WARN] Could not ensure RECORD_ROOT on NAS:", e)
    # Camera init
    cam_src = int(os.getenv("CAMERA_SRC", 5))
    vs = VideoStream(src=cam_src).start()
    time.sleep(1.0)
    # Warmup read
    for _ in range(50):
        frame = vs.read()
        if frame is not None:
            break
        time.sleep(0.05)
    if frame is None:
        raise RuntimeError("Camera did not produce a frame. Check CAMERA_SRC.")

    h0, w0 = frame.shape[:2]
    new_w = int(w0 * SCALE_FACTOR)
    new_h = int(h0 * SCALE_FACTOR)

    cv2.namedWindow("Real‑Time Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Real‑Time Detection", new_w, new_h)
    cv2.setWindowProperty("Real‑Time Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Recording state
    segment_writer = None
    segment_start_ts = None
    segment_path = None
    segment_had_skip = False
    prev_state = None
    frame_rate_calc = 0.0
    freq = cv2.getTickFrequency()
    is_skip_active = False
    prev_alert_active = False

    while True:
        t1 = cv2.getTickCount()
        frame = vs.read()
        if frame is None:
            break

        detected_objects = set()

        with interpreter_lock:
            th = model_thresholds["model"]
            inp = input_details
            out = output_details
            intr = interpreter

        if intr is None:
            raise RuntimeError("Interpreter not initialized")

        input_data, meta = preprocess_frame(frame, inp)
        intr.set_tensor(inp[0]["index"], input_data)
        intr.invoke()

        out_data = intr.get_tensor(out[0]["index"])
        out_scale, out_zp = out[0]["quantization"]
        boxes, scores, class_ids = process_yolov5_output(
            out_data, meta, (out_scale, out_zp), th["DETECTION_THRESHOLD"]
        )
        boxes, scores, class_ids = apply_nms_per_class(
            boxes, scores, class_ids, th["IOU_THRESHOLD"], th["SCORE_THRESHOLD"]
        )

        # Draw detections and collect class names
        local_cfg = cfg  # snapshot under lock is OK since replaced wholesale
        for i, box in enumerate(boxes):
            xmin, ymin, xmax, ymax = box
            cid = class_ids[i]
            score = scores[i]
            class_name = local_cfg.idx_to_name.get(cid + 1, str(cid))
            draw_detection(frame, xmin, ymin, xmax, ymax, class_name, score)
            if class_name != "Unknown":
                detected_objects.add(class_name)

        # Build rt vector
        rt = -np.ones(local_cfg.n_classes, dtype=np.int8)
        for name in detected_objects:
            if name in local_cfg.name_to_idx:
                rt[local_cfg.name_to_idx[name]] = 1

        current_state = sm.update(rt)
        detection_status = sm.build_status_dict(rt)
        # Trigger an alert (sound + overlay list) on first skipped event
        if any(s == "Skipped" for s in detection_status.values()):
            if not is_skip_active:
                trigger_skip_alert(detection_status, duration=3.0)
            is_skip_active = True
            segment_had_skip = True
        else:
            is_skip_active = False

        draw_status_list(frame, detection_status, current_state, objects, local_cfg)

        render_missing_alert(frame)
        # Stop recording when the alarm overlay window finishes
        alert_active_now = (time.time() < alert_end_time)
        if prev_alert_active and not alert_active_now:
            # Alarm just ended → finalize current segment if it had a skip
            if segment_writer is not None and segment_had_skip:
                segment_writer.release()
                _append_skip_log(segment_start_ts, segment_path)
                print(f"[RECORD] Saved (skip, alarm end): {segment_path}")
                segment_writer = None
                segment_path = None
                segment_had_skip = False
        prev_alert_active = alert_active_now
        # HUD
        hud = frame.copy()
        cv2.putText(hud, f"FPS: {frame_rate_calc:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(hud, f"Model: {Model_ID}", (250, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        det_t = model_thresholds["model"]["DETECTION_THRESHOLD"]
        iou_t = model_thresholds["model"]["IOU_THRESHOLD"]
        sco_t = model_thresholds["model"]["SCORE_THRESHOLD"]
        cv2.putText(hud, f"Det: {det_t:.2f} | IoU: {iou_t:.2f} | Score: {sco_t:.2f}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Write to active recording (full-res HUD)
        if segment_writer is not None:
            segment_writer.write(hud)

        # Segment boundary: when we ENTER the end-state (e.g., Box Closing)
        entered_end_state = (current_state == END_STATE_RESOLVED and prev_state != END_STATE_RESOLVED)
        if entered_end_state:
            # finalize previous segment (between last end-state and now)
            if segment_writer is not None:
                segment_writer.release()
                if segment_had_skip:
                    _append_skip_log(segment_start_ts, segment_path)
                else:
                    try:
                        os.remove(str(segment_path))
                    except Exception as _e:
                        print("[WARN] Couldn't delete clean segment:", _e)
                segment_writer = None
                segment_path = None

            # start a new segment beginning at this end-state
            segment_start_ts = time.time()
            fname = f"{line_id}_{Model_ID}_{time.strftime('%Y%m%d_%H%M%S', time.localtime(segment_start_ts))}{RECORD_EXT}"
            segment_path = RECORD_ROOT / fname
            fps_i = int(round(frame_rate_calc)) if frame_rate_calc > 1 else 20
            fps_i = max(5, min(60, fps_i))
            fourcc = cv2.VideoWriter_fourcc(*RECORD_CODEC)
            h_rec, w_rec = hud.shape[0], hud.shape[1]
            segment_writer = cv2.VideoWriter(str(segment_path), fourcc, float(fps_i), (w_rec, h_rec))
            if not segment_writer.isOpened():
                print("[ERROR] Failed to open video writer at", segment_path)
            else:
                print(f"[RECORD] Start: {segment_path}")
            segment_had_skip = False

        prev_state = current_state

        resized = cv2.resize(hud, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        cv2.imshow("Real‑Time Detection", resized)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("+"):
            model_thresholds["model"]["DETECTION_THRESHOLD"] = min(det_t + 0.05, 1.0)
        elif key == ord("-"):
            model_thresholds["model"]["DETECTION_THRESHOLD"] = max(det_t - 0.05, 0.0)
        elif key == ord(","):
            model_thresholds["model"]["IOU_THRESHOLD"] = min(iou_t + 0.05, 1.0)
        elif key == ord("."):
            model_thresholds["model"]["IOU_THRESHOLD"] = max(iou_t - 0.05, 0.0)
        elif key == ord("["):
            model_thresholds["model"]["SCORE_THRESHOLD"] = min(sco_t + 0.05, 1.0)
        elif key == ord("]"):
            model_thresholds["model"]["SCORE_THRESHOLD"] = max(sco_t - 0.05, 0.0)
            
        t2 = cv2.getTickCount()
        time1 = (t2 - t1) / freq
        if time1 > 0:
            frame_rate_calc = 1.0 / time1

    # Finalize any open segment on exit
    try:
        if segment_writer is not None:
            segment_writer.release()
            if segment_had_skip:
                _append_skip_log(segment_start_ts, segment_path)
            else:
                try:
                    os.remove(str(segment_path))
                except Exception as _e:
                    print("[WARN] Couldn't delete clean segment:", _e)
    finally:
        vs.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
