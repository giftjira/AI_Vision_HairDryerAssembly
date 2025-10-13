# AI Vision – Hair Dryer Assembly Monitoring

> Real‑time computer vision for monitoring a hair‑dryer assembly/packing line: track cycle time, detect missing parts, and alert operators.

---

## Overview
This project leverages image **classification** and **object detection** to monitor production processes in real time. It tracks cycle times, detects missing parts, and ensures accuracy in assembly and packing processes on the line hardware (Tinker Board + USB camera, alarm speaker, and monitor).

---

## Project structure
```
AI_Vision_HairDryerAssembly/
├─ CCTV_AssemblyLine/           # Camera capture & on‑line inference scripts
├─ CycleTimeTracking/           # Logic for timing cycles via part‑present/absent transitions
├─ PartMissingDetection/        # Object detection / classification models & runtime
├─ AssemblyLineVisualization/   # Simple visualization of line status & KPIs
├─ Dataset Processing Tool/     # Tools for curation/augmentation/format conversion of datasets
├─ Hardware Setup/              # Notes/scripts for Tinker Board + peripherals
└─ README.md
```

> Names mirror the folders in this repo. If you later rename or add modules, update the table above accordingly.

---

## How it works

### Pipeline at a glance
```mermaid
flowchart LR
    A[USB Camera] --> B[Frame Capture]
    B --> C[Preprocessing\n(resize, ROI, normalization)]
    C --> D{Model}
    D -->|Object detection| E[Part Bounding Boxes & Classes]
    D -->|Classification| F[Present / Absent score]
    E --> G[State Machine\nper station]
    F --> G
    G --> H[Cycle Time Calculator]
    G --> I[Missing‑Part Alert]
    H --> J[CSV/DB Logger]
    I --> K[Alarm Speaker & On‑screen overlay]
    J --> L[Visualization / Dashboard]
```

### Core ideas
- **Sensing**: A USB camera watches each work area. Frames are sampled at a modest FPS (e.g., 10–20) to balance latency and CPU load.
- **Inference**: A lightweight detector (e.g., MobileNet‑SSD, YOLO‑Nano) or a classifier trained per station determines whether each required part is **present**.
- **State machine**: For each station, a tiny state machine consumes the present/absent signal to detect **start** → **in‑progress** → **complete** transitions. That enables **per‑unit cycle time** without hardware PLC taps.
- **Missing‑part logic**: When a required class is absent at a step where it should appear, or when an expected dwell exceeds a threshold, the system raises an on‑screen banner and triggers an **audible alarm**.
- **Logging & analytics**: Timestamps and station events are appended to CSV files (or a small DB). The visualization module renders key KPIs: current cycle, last N cycles, and counts of alerts.

### Training & dataset flow
1. **Collect** images/video on the line (use `CCTV_AssemblyLine/` capture utilities).
2. **Curate/annotate** with your preferred tool (Roboflow, Teachable Machine). Export as COCO/YOLO as needed.
3. **Convert/augment** with utilities in `Dataset Processing Tool/`.
4. **Train** tiny models in Google Colab for portability to the Tinker Board.
5. **Export** weights and place them in `PartMissingDetection/` (keep versioned subfolders by date/model name).

### Runtime
- **On device**: Tinker Board running Debian starts the capture + inference script at boot (systemd or simple shell wrapper).
- **Outputs**: live window/HDMI overlay for operators, periodic CSV logs, and optional beeper/speaker alarms.

---

## Getting started
### Prerequisites
- **Python** 3.9+ on your dev machine (VS Code recommended).
- **Libraries**: OpenCV, PyTorch *or* TensorFlow/Keras (choose based on your trained model), NumPy, Pandas.
- **Hardware**: Tinker Board (Debian), USB camera, speaker/buzzer, HDMI display.

> Tip: If you use TensorFlow on ARM, prefer TF‑Lite builds for better performance.

### Setup (development)
```bash
# 1) Clone
git clone https://github.com/giftjira/AI_Vision_HairDryerAssembly.git
cd AI_Vision_HairDryerAssembly

# 2) Create a virtual env (example with Python 3.10)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3) Install the essentials (adapt as needed)
pip install opencv-python numpy pandas
# Choose ONE of the two families:
#   PyTorch:   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
#   TensorFlow: pip install tensorflow  # or tflite-runtime on ARM
```

### Run (example)
```bash
# Example: start camera + inference + simple overlay
python CCTV_AssemblyLine/run.py \
  --model PartMissingDetection/models/yolo_nano_best.onnx \
  --labels PartMissingDetection/labels.txt \
  --roi configs/station1_roi.yaml \
  --output logs/station1.csv
```
> Adjust paths/flags to match your model files and ROI configs. If your entrypoints differ, update the command here to your actual script names.

---

## Configuration
- **ROI per station**: YAML/JSON file that pins the camera area for each work step.
- **Thresholds**: probability thresholds per class and **dwell limits** (seconds) before triggering an alert.
- **Logging**: path pattern for CSV files and retention window.

> Keep all tunables in a `configs/` folder so ops can tweak without editing code.

---

## Visualization & KPIs
- Current unit cycle time and rolling mean (e.g., last 20 units)
- Count of missing‑part alerts by station
- Uptime / frames processed / FPS

A minimal web/desktop viewer in `AssemblyLineVisualization/` can read the CSVs and plot these summaries. Consider adding a small Flask app for a live dashboard.

---

## Repository roadmap (ideas)
- [ ] Add `requirements.txt` and/or `pyproject.toml`
- [ ] Provide sample configs and a tiny sample clip
- [ ] Export a pre‑trained demo model with fake data
- [ ] Add systemd service file for auto‑start on Tinker Board
- [ ] Unit tests for state‑machine logic

---

## License
Add a license file (MIT/Apache‑2.0) if you’re planning to open‑source usage.

---

## Acknowledgements
- Roboflow, Google Colab, and Teachable Machine were used to prepare and train models.

---

*This README replaces the short overview with a runnable, documented structure and a clear "How it works" section. If you maintain separate entry points per module, update the commands above to point to your actual scripts.*

