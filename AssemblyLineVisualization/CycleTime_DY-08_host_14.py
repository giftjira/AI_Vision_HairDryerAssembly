# ---------------------------------------------------------------------------
# DESIGN OVERVIEW
# ---------------------------------------------------------------------------
# Layout Grid (matplotlib.gridspec 3x4):
#   Row0: Header across all 4 columns.
#   Row1: All-Day Trend (cols 0-1), Real-Time Bar/KPI (cols 2-3).
#   Row2: Live Stream w/ overlays (cols 0-1), Production Data table (col 2),
#         Statistic Data table (col 3).
#
# ---------------------------------------------------------------------------
# CODE BEGINS
# ---------------------------------------------------------------------------

from __future__ import annotations
import cv2
import threading
import time
import os
import sys
import csv
import math
import queue
import logging
import argparse
import subprocess
import webbrowser
import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple , Set

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, TextBox ,CheckButtons
from matplotlib.lines import Line2D
from matplotlib.collections import PathCollection

try:
    from matplotlib.widgets import Dropdown  # some builds don't ship it
except Exception:
    Dropdown = None

# Optional imports (guarded) -------------------------------------------------
try:
    import paho.mqtt.client as mqtt  # type: ignore
except Exception:  # pragma: no cover - if not installed
    mqtt = None

try:
    import pyodbc  # type: ignore
except Exception:  # pragma: no cover
    pyodbc = None

try:
    from PIL import Image, ImageTk
except Exception:  # pragma: no cover
    Image = None

# ----------------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------------
global mp
DATA_DIR = Path(".")
filename = os.path.basename(__file__)
LINE_DEFAULT = filename.split('_')[1]
lines = LINE_DEFAULT.split('-')[0] + LINE_DEFAULT.split('-')[1]

global current_date
current_date = dt.date.today()
VIDEO_DIR_DEFAULT = Path("C:/Videos") if os.name == "nt" else Path("./videos")
csv_filename = f"data_{lines}_{current_date}.csv"

DB_DRIVER = "SQL Server"
DB_SERVER = "158.118.37.201" 
DB_NAME   = "PMFTH-PMS_P2"
DB_USER   = "PMFMES"
DB_PASS   = "Str0ngP@ssw0rd!"

conn_str = (
    f"DRIVER={{{DB_DRIVER}}};"
    f"SERVER={DB_SERVER};"
    f"DATABASE={DB_NAME};"
    f"UID={DB_USER};PWD={DB_PASS}"
)

# Shift windows used to color the all-day chart background. Adjust per site.
SHIFT_SEGMENTS = [
    (dt.time(7, 30), dt.time(9, 30), "AM"),
    (dt.time(9, 30), dt.time(9, 40), "BREAK"),
    (dt.time(9, 40), dt.time(11, 30), "AM"),
    (dt.time(11, 30), dt.time(12, 20), "LUNCH"),
    (dt.time(12, 20), dt.time(14, 30), "PM"),
    (dt.time(14, 30), dt.time(14, 40), "BREAK"),
    (dt.time(14, 40), dt.time(16, 20), "PM"),
]

SHIFT_PERIOD = {
    '1': ["07:30","09:30"],
    '2': ["09:40","11:30"],
    '3': ["12:20","14:30"],
    '4': ["14:40","16:20"],
    '5': ["16:50","19:20"]
}

def period_tk():
    now = dt.datetime.now().time()
    for shift, (start_str, end_str) in SHIFT_PERIOD.items():
        start_time = dt.datetime.strptime(start_str, "%H:%M").time()
        end_time = dt.datetime.strptime(end_str, "%H:%M").time()
        if start_time <= now <= end_time:
            return shift  # return as integer 1–5
    return None  # no shift matches

REFRESH_UI_SEC = 10
REFRESH_DB_SEC = 10
LIVE_STREAM_SRC = "rtsp://10.84.171.19:8554/mystream"
USE_DSHOW = True
# KPI thresholds --------------------------------------------------------------
# STD target vs Upper Limit (UL) factor. Example from screenshot: STD 70, UL +20% = 84.
UL_FACTOR = 1.50
# Acceptable fluctuation (%) threshold for green check vs red X.
FLUCT_OK_PCT = 10.0

# --- StepTracking (NAS) ---
STEPTRACK_ROOT      = r"\\158.118.87.150\pmfth-nas\CameraCapture-AI-Vision"
STEPTRACK_SKIPS_DIR = os.path.join(STEPTRACK_ROOT, "Skipped_Video")
STEPTRACK_VIDEO_DIR = STEPTRACK_SKIPS_DIR
def _skip_log_path_for(date_: dt.date) -> str:
    # file name pattern: skip_log_YYYYMMDD.csv
    return os.path.join(STEPTRACK_SKIPS_DIR, f"skip_log_{date_:%Y%m%d}.csv")


# Logging --------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("dy08")

# ----------------------------------------------------------------------------
# DATA CLASSES
# ----------------------------------------------------------------------------

@dataclass
class Sample:
    ts: dt.datetime
    ct: float  # cycle time sec

@dataclass
class ProcessSeries:
    pid: str
    samples: List[Sample] = field(default_factory=list)

    def add(self, ts: dt.datetime, ct: float) -> None:
        self.samples.append(Sample(ts, ct))

    # Metrics ----------------------------------------------------------------
    @property
    def values(self) -> np.ndarray:
        if not self.samples:
            return np.empty(0)
        return np.array([v.ct for v in self.samples if v.ct < 600], dtype=float)
        
    def last(self) -> Optional[Sample]:
        return self.samples[-1] if self.samples else None

    def avg(self) -> Optional[float]:
        v = self.values
        return float(np.mean(v)) if v.size else None

    def avgs(self) -> Optional[float]:
        v = self.values
        if not v.size:
            return None
        last10 = v[-5:] if v.size > 5 else v
        return float(np.mean(last10))
    
    def ct_mins(self) -> Optional[float]:
        v = self.values
        if not v.size:
            return None
        last10 = v[-5:] if v.size > 5 else v
        return float(np.min(last10))
    
    def ct_maxs(self) -> Optional[float]:
        v = self.values
        if not v.size:
            return None
        last10 = v[-5:] if v.size > 5 else v
        return float(np.max(last10))
    
    def fluct_pcts(self) -> Optional[float]:
        v = self.values
        last10 = v[-5:] if v.size > 5 else v
        if last10.size < 2:
            return 0.0 if last10.size else None
        mn, mx = float(np.min(last10)), float(np.max(last10))
        av = float(np.mean(last10)) or 1.0
        return (mx - mn) / av * 100.0

    def ct_min(self) -> Optional[float]:
        v = self.values
        return float(np.min(v)) if v.size else None

    def ct_max(self) -> Optional[float]:
        v = self.values
        return float(np.max(v)) if v.size else None

    def fluct_pct(self) -> Optional[float]:
        v = self.values
        if v.size < 2:
            return 0.0 if v.size else None
        mn, mx = float(np.min(v)), float(np.max(v))
        av = float(np.mean(v)) or 1.0
        return (mx - mn) / av * 100.0

    def values_before(self, t: dt.datetime) -> np.ndarray:
        """All ct values with timestamp <= t (and <600 filter)."""
        if not self.samples:
            return np.empty(0)
        vals = [s.ct for s in self.samples if s.ts <= t and s.ct < 600]
        return np.array(vals, dtype=float)

    def last_n_before(self, t: dt.datetime, n: int = 5) -> List[Sample]:
        """Last n samples with timestamp <= t."""
        if not self.samples:
            return []
        arr = [s for s in self.samples if s.ts <= t]
        return arr[-n:]

        
        

@dataclass
class ProductionPeriod:
    start: dt.datetime  ######
    end: dt.datetime ######
    model: str ######
    status: str  # e.g., "End Production", "On Production" #######
    plan_qty: Optional[int] 
    actual_qty: Optional[int] 
    std_ct: Optional[float] 
    ct_min: Optional[float]
    ct_max: Optional[float] 
    avg_ct: Optional[float] 
    fluct_pct: Optional[float]

@dataclass
class DashboardState:
    line: str 
    n_proc: int # Number of processes (P1, P2, ..., Pn)
    date: dt.date  # Default to today
    std_time: Optional[float] = 0.0  # target cycle time
    ul_time: Optional[float] = 0.0   # upper limit (std * UL_FACTOR)
    processes: Dict[str, ProcessSeries] = field(default_factory=dict)
    periods: List[ProductionPeriod] = field(default_factory=list)
    live_frame_path: Optional[Path] = None

    def __post_init__(self) -> None:
        if not self.processes:
            for i in range(1, self.n_proc + 1):
                self.processes[f"P{i}"] = ProcessSeries(pid=f"P{i}")
            self.processes["TT"] = ProcessSeries(pid="TT")
        if self.std_time is not None and self.ul_time is None:
            self.ul_time = self.std_time * UL_FACTOR

    # Convenience -------------------------------------------------------------
    def proc_ids(self) -> List[str]:
        return list(self.processes.keys())

    def proc(self, pid: str) -> ProcessSeries:
        return self.processes[pid]

    def all_samples(self) -> List[Sample]:
        out: List[Sample] = []
        for p in self.processes.values():
            out.extend(p.samples)
        return out

# ----------------------------------------------------------------------------
# DATABASE ACCESS (SAFE WRAPPERS)
# ----------------------------------------------------------------------------

Q_TT = (
    "SELECT ProductionDate, LineId, TaskNo, PlanQty, Status , ActualStartOperationDateTime, ActualEndOperationDateTime ,CreatedByUserId FROM TT_ProductionPlan "
    "WHERE ProductionDate = ? AND LineId = ? AND Status = 'On Production';"
)
Q_XT = (
    "SELECT ModelId FROM XT_ProductionPlan "
    "WHERE ProductionDate = ? AND LineId = ? AND TaskNo = ?;"
)
Q_MANPOWER = (
    "SELECT ManPower FROM XT_ProductionPlan "
    "WHERE ProductionDate = ? AND LineId = ? AND ModelId = ?;"
)
Q_STDTIME = "SELECT StdTime FROM GFN_STDTimeAllData WHERE Model = ?;"

Q_LEADER = "SELECT UserName FROM TM_User WHERE UserId = ?"

Q_ACTUAL = "SELECT ActualQty FROM TT_ProductionActual WHERE ProductionDate = ? AND LineId = ? AND TaskNo = ?"

Q_TK = "SELECT UpdatedDateTime  FROM TT_ProductionActual WHERE ProductionDate = ? AND LineId = ? AND TaskNo = ? AND PeriodId = ?"

@dataclass
class DBConfig:
    driver: str 
    server: str 
    name: str 
    user: str 
    password: str 

    def conn_str(self) -> str:
        return (
            f"DRIVER={{{self.driver}}};SERVER={self.server};DATABASE={self.name};"
            f"UID={self.user};PWD={self.password}"
        )

def fetch_model_id(line: str, date_: dt.date) -> Optional[str]:
    if pyodbc is None:
        print('cant connect')
        return None
    model = ''
    try:
        with pyodbc.connect(conn_str) as conn:
            cur = conn.cursor()
            cur.execute(Q_TT, (date_.strftime('%Y-%m-%d'), line))
            rows = cur.fetchall()
            for prd, lid, tno,plan, status,aod,aed,cbid in rows:
                cur.execute(Q_XT, (prd, lid, tno))
                r = cur.fetchone()
                if r and getattr(r, "ModelId", None):
                    model=str(r.ModelId).strip()
                    return rows ,model
    except Exception as exc:
        log.exception("fetch_model_id failed: %s", exc)
    return None

def fetch_leader(line: str, date_: dt.date) -> Optional[str]:
    if pyodbc is None:
        return None
    try:
        with pyodbc.connect(conn_str) as conn:
            cur = conn.cursor()
            cur.execute(Q_TT, (date_.strftime('%Y-%m-%d'), line))
            rows = cur.fetchall()
            for prd, lid, tno,plan, status,aod,aed,cbid in rows:
                cur.execute(Q_LEADER, (cbid))
                r = cur.fetchone()
                if r and getattr(r, "UserName", None):
                    return str(r.UserName).strip()
    except Exception as exc:
        log.exception("fetch_leader failed: %s", exc)
    return None

def fetch_std_time(model: str, line: str, date_: dt.date) -> Tuple[Optional[float], Optional[float]]:
    """Return (std_time, manpower_adjusted) where manpower_adjusted = std/manpower."""
    if pyodbc is None or model is None:
        return (None, None)
    std_t: Optional[float] = None
    mp_adj: Optional[float] = None
    mp: Optional[int] = None
    try:
        with pyodbc.connect(conn_str) as conn:
            cur = conn.cursor()
            cur.execute(Q_STDTIME, model)
            r = cur.fetchone()
            if r and getattr(r, "StdTime", None) is not None:
                std_t = float(r.StdTime)
            cur.execute(Q_MANPOWER, (date_.strftime('%Y-%m-%d'), line, model))
            rs = cur.fetchone()
            if rs and getattr(rs, "ManPower", None) not in (None, 0):
                mp = int(rs.ManPower)
                if std_t is not None and mp > 0:
                    mp_adj = std_t / mp
                    return (std_t, mp_adj, mp)
    except Exception as exc:
        log.exception("fetch_std_time failed: %s", exc)
    return None

def fetch_actual(line: str, date_: dt.date) -> Optional[int]:
    if pyodbc is None:
        print('cant connect')
        return None
    act = None
    try:
        with pyodbc.connect(conn_str) as conn:
            cur = conn.cursor()
            cur.execute(Q_TT, (date_.strftime('%Y-%m-%d'), line))
            rows = cur.fetchall()
            for prd, lid, tno,plan, status,aod,aed,cbid in rows:
                cur.execute(Q_ACTUAL, (prd, lid, tno))
                r = cur.fetchall()
                act = sum(r[0] for r in r)
                return act
    except Exception as exc:
        log.exception("fetch_actual failed: %s", exc)
    return None

def fetch_takt_time(line: str, date_: dt.date) -> Optional[float]:
    try:
        with pyodbc.connect(conn_str) as conn:
            cur = conn.cursor()
            cur.execute(Q_TT, (date_.strftime('%Y-%m-%d'), line))
            rows = cur.fetchall()
            for prd, lid, tno,plan, status,aod,aed,cbid in rows:
                cur.execute(Q_TK, (prd, lid, tno,period_tk()))
                r = cur.fetchone()
                if r and getattr(r, "UpdatedDateTime", None):
                    t = r.UpdatedDateTime
                    if isinstance(t, dt.datetime):
                        return t
    except Exception as exc:
        log.exception("fetch_actual failed: %s", exc)
    return None

models=fetch_model_id(LINE_DEFAULT,current_date)[1]
mp = fetch_std_time(models,LINE_DEFAULT, current_date)[2]
std_time = fetch_std_time(models,LINE_DEFAULT,current_date)[1]

# ----------------------------------------------------------------------------
# MQTT INGEST (THREAD-SAFE QUEUE)
# ----------------------------------------------------------------------------

@dataclass
class MQTTConfig:
    host: str = "10.84.171.108"
    port: int = 1883
    keepalive: int = 60

start_time = dt.time(7, 30)
end_time = dt.time(19, 30)

class MQTTIngest:
    """Subscribe to line-specific process topics and push samples into a queue."""

    def __init__(self, cfg: MQTTConfig, line: str, out_q: queue.Queue):
        self.cfg = cfg
        self.line = line
        self.out_q = out_q
        self.client = None
        if mqtt is not None:
            self.client = mqtt.Client(protocol=mqtt.MQTTv311)
            self.client.on_connect = self._on_connect
            self.client.on_message = self._on_message

    def start(self) -> None:
        if self.client is None:
            log.warning("MQTT not available; running w/o live messages")
            return
        try:
            self.client.connect(self.cfg.host, self.cfg.port, self.cfg.keepalive)
            self.client.publish(f"{lines}/Model_ID",payload=f"{models}_{std_time}", retain = True)
            self.client.loop_start()
        except Exception as exc:
            log.exception("MQTT connect failed: %s", exc)

    def stop(self) -> None:
        if self.client is None:
            return
        try:
            self.client.loop_stop()
        except Exception:  # pragma: no cover
            pass

    # MQTT callbacks ---------------------------------------------------------
    def _on_connect(self, client, userdata, flags, rc):  # pragma: no cover - network
        if rc != 0:
            log.error("MQTT connect rc=%s", rc)
            return
        for i in range(1, mp + 1):
            topic = f"{self.line}_P{i}/cycle_time"
            client.subscribe(topic)
            log.info("Subscribed %s", topic)
        client.subscribe(f"{self.line}/StepTracking")

    def _on_message(self, client, userdata, msg):  # pragma: no cover - network
        try:
            now = dt.datetime.now()
            now = now.__format__('%H:%M:%S')
            for (st, et, lbl) in SHIFT_SEGMENTS:
                if lbl == "BREAK" or lbl == "LUNCH":
                    if str(st) <= now <= str(et):
                        log.info("Skipping message during %s segment: %s", lbl, msg.topic)
                        return    
            payload = msg.payload.decode()
            print("Topic ",msg.topic,"CycleTime ", payload)
        except Exception as exc:
            log.warning("Bad MQTT message %s: decode error: %s", msg.topic, exc)
            return

        try:
            ct = float(payload)
        except Exception as exc:
            log.warning("Bad MQTT message %s: cannot convert payload '%s' to float: %s", msg.topic, payload, exc)
            return

        topic = msg.topic
        topic_parts = topic.split('/')
        if len(topic_parts) < 2:
            return

        process_id_part = topic_parts[0].split('_')
        if len(process_id_part) < 2:
            return

        process_id = process_id_part[-1]
        now = dt.datetime.now()
        current_time = now.time()
        if start_time <= current_time <= end_time:
            if "cycle_time" in topic:
                append_csv(process_id,now,ct,csv_filename)

        self.out_q.put((process_id, now, ct))

# ----------------------------------------------------------------------------
# LiveStream
# ----------------------------------------------------------------------------

class LiveStream:
    """
    Background frame grabber so the dashboard refresh is smooth.
    Accepts webcam index (int) or path/URL (str).
    """
    def __init__(self, src, reopen_delay=2.0):
        self.src = src
        self.reopen_delay = reopen_delay
        self.cap = None
        self.frame = None
        self._lock = threading.Lock()
        self._run = False
        self._t = None

    def get_frame(self):
        return self.frame
    
    def start(self):
        if self._run:
            return
        self._run = True
        self._t = threading.Thread(target=self._loop,args=(), daemon=True)
        self._t.start()

    def stop(self):
        self._run = False
        if self._t:
            self._t.join(timeout=1)
        if self.cap:
            self.cap.release()
            self.cap = None

    def _open_cap(self):
    # Accept int (USB cam index) or str (rtsp/http/file)
        if isinstance(self.src, int):
            if USE_DSHOW and hasattr(cv2, "CAP_FFMPEG"):
                cap = cv2.VideoCapture(self.src)
            else:
                cap = cv2.VideoCapture(self.src)
        else:
            cap = cv2.VideoCapture(self.src)
        if not cap.isOpened():
            return None
        return cap

    def _loop(self):
        while self._run:
            if self.cap is None:
                self.cap = self._open_cap()
                if self.cap is None:
                    time.sleep(self.reopen_delay)
                    continue
            ok, frm = self.cap.read()
            if not ok:
                # lost connection; drop & retry
                self.cap.release()
                self.cap = None
                time.sleep(self.reopen_delay)
                continue
            with self._lock:
                self.frame = frm

    def latest(self):
        with self._lock:
            if self.frame is None:
                return None
            return self.frame.copy()

# ----------------------------------------------------------------------------
# DASHBOARD RENDERER
# ----------------------------------------------------------------------------
def fetch_steptracking_skipped_for_date(d: dt.date) -> List[Tuple[dt.datetime, str, str]]:
    csv_path = _skip_log_path_for(d)
    out: List[Tuple[dt.datetime, str, str]] = []
    if not os.path.exists(csv_path):
        log.warning("StepTracking log not found: %s", csv_path)
        return out
    try:
        df = pd.read_csv(csv_path, dtype=str, engine="python", on_bad_lines="skip")
        cols = {c.strip().lower(): c for c in df.columns}
        if "time_error" not in cols or "record_filename" not in cols:
            log.warning("StepTracking CSV missing columns; have=%s", list(df.columns))
            return out
        te_col, fn_col = cols["time_error"], cols["record_filename"]
        for _, row in df.iterrows():
            ts = _parse_time_error(str(row.get(te_col, "")))
            fn = os.path.basename(str(row.get(fn_col, "")).strip())
            if ts and fn:
                out.append((ts, fn, os.path.join(STEPTRACK_VIDEO_DIR, fn)))
    except Exception as exc:
        log.exception("Failed reading StepTracking CSV %s: %s", csv_path, exc)
    out.sort(key=lambda r: r[0], reverse=True)
    return out

def _parse_time_error(s: str) -> Optional[dt.datetime]:
    s = (s or "").strip()
    if not s or s.startswith("#"):
        return None
    for fmt in ("%m/%d/%Y %I:%M:%S %p", "%m/%d/%Y %I:%M %p"):
        try:
            return dt.datetime.strptime(s, fmt)
        except ValueError:
            pass
    return None
    
def load_steptracking_csv_for_date(d: dt.date) -> List[Dict]:
    """
    Reads StepTracking 'skip_log_YYYYMMDD.csv' from NAS and returns a list of dicts:
      { 'ts': datetime, 'filename': 'DY08_....mp4', 'fullpath': '\\\\nas\\...\\DY08_....mp4' }
    Rows with bad/missing timestamps are skipped.
    """
    csv_path = _skip_log_path_for(d)
    out: List[Dict] = []
    if not os.path.exists(csv_path):
        logging.warning("StepTracking CSV does not exist: %s", csv_path)
        return out

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        cols = [c.strip().lower() for c in (reader.fieldnames or [])]
        # Normalize keys to lower for access
        for row in reader:
            # tolerate header variations
            tkey = "time_error" if "time_error" in cols else "timestamp"
            fkey = "record_filename" if "record_filename" in cols else "filename"

            raw_ts = row.get(tkey, "")
            raw_fn = row.get(fkey, "")
            ts = _parse_time_error(raw_ts)
            if ts is None:
                continue
            fname = os.path.basename(raw_fn or "").strip()
            if not fname:
                continue
            fullpath = os.path.join(STEPTRACK_VIDEO_DIR, fname)
            out.append({"ts": ts, "filename": fname, "fullpath": fullpath})
    # newest first
    out.sort(key=lambda r: r["ts"], reverse=True)
    return out

class DY08Dashboard:
    def __init__(self, state: DashboardState, data_dir, video_dir: Path, livestream: LiveStream):
        self.state = state
        self.data_dir = data_dir
        self.video_dir = video_dir
        self.livestream = livestream
        self.ltk = None
        self.diff_seconds = 0.0
        self._sel_time: Optional[dt.datetime] = None

        self.fig = plt.figure(figsize=(16, 9), dpi=110, constrained_layout=False)
        try:
            self.fig.canvas.manager.set_window_title(f"{state.line} Dashboard")
        except Exception:
            pass

        gs = self.fig.add_gridspec(
            14, 12,   # bump rows to 14 (more control)
            left=0.055, right=0.985, top=0.965, bottom=0.055,
            wspace=0.55, hspace=0.38
        )
        self.ax_header    = self.fig.add_subplot(gs[0:2, 2:10])
        self.ax_step_btn  = self.fig.add_subplot(gs[0, 1:2])
        self.ax_button    = self.fig.add_subplot(gs[0, 0:1])
        self.ax_drop_date = self.fig.add_subplot(gs[0,10:12])

        # --- Top row content ---
        self.ax_all_day   = self.fig.add_subplot(gs[2:7, 0:7])   # taller all-day plot
        self.ax_prod_tbl  = self.fig.add_subplot(gs[2:3, 7:12])  # very short row
        self.ax_stats_tbl = self.fig.add_subplot(gs[4:6, 7:12])  # compact stats

        # --- Bottom row content ---
        self.ax_rt_bar    = self.fig.add_subplot(gs[8:14, 0:7])  # taller RT bar
        self.ax_fluc      = self.ax_rt_bar.twinx()
        self.ax_live      = self.fig.add_subplot(gs[6:14, 7:12]) # live fills rest
        self.ax_live.axis("off")

        prop_cyc = plt.rcParams['axes.prop_cycle'].by_key()['color']
        self.bbox_all_day = self.ax_all_day.get_position()
        self.proc_colors: Dict[str, str] = {}
        for i, pid in enumerate(self.state.proc_ids()):
            self.proc_colors[pid] = prop_cyc[i % len(prop_cyc)]

        self.last_db_refresh = dt.datetime.min
        self.pause = False

        # keyboard
        self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self._on_key)

        # -------- Picking/highlight state --------
        self.series_visible = {pid: True for pid in self.state.proc_ids()}  # pid -> bool
        self.line_handles: Dict[str, Line2D] = {}                           # pid -> line
        self.point_handles: Dict[str, PathCollection] = {}                  # pid -> scatter
        self._series_cache: Dict[str, Tuple[List[dt.datetime], List[float]]] = {}  # pid -> (t, v)
        self.legend_all_day = None

        self._hl_sel = None               # {"pid": str, "end_idx": int}
        self._hl_artists: List = []       # overlay artists to remove

        self.show_rt_values = False
        self._rt_bar_labels = []      # list[matplotlib.text.Text]
        self._rt_avg_plot_label = []
        self._rt_max_plot_label = []   
        self._rtbar_ck_ax = None      # small axes for the checkbox
        self._rtbar_ck = None

        # Single pick handler (legend + points)
        self.cid_pick = self.fig.canvas.mpl_connect('pick_event', self._on_pick_all_day)
        # Extra: mouse-click handler to toggle legend items (more reliable than pick_event for legends)
        self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self._on_click_legend)

    # ---------- events ----------
    def _on_key(self, event):
        if event.key == ' ':
            self.pause = not self.pause
        elif event.key == 'r':
            self._force_refresh_db()
        elif event.key == 'q':
            plt.close(self.fig)

    def refresh(self, db: Optional[DBConfig] = None) -> None:
        if self.pause:
            return
        now = dt.datetime.now()
        load_csv(self.state, self.data_dir)

        if db is not None and (now - self.last_db_refresh).total_seconds() >= REFRESH_DB_SEC:
            self._refresh_db()
            self.last_db_refresh = now

        self._draw_header()
        self._draw_all_day()
        self._ensure_rtbar_checkbox()
        self._draw_rt_bar()     
        self._draw_button()
        self._draw_drop_date()
        self._draw_prod_tbl()
        self._draw_stats_tbl()
        self.fig.canvas.draw_idle()

    def _refresh_db(self) -> None:
        global mp
        self.prod_data ,self.model = fetch_model_id(self.state.line, self.state.date)
        self.leader = fetch_leader(self.state.line, self.state.date)
        self.actual = fetch_actual(self.state.line, self.state.date)
        self.tt = fetch_takt_time(self.state.line, self.state.date)
        if self.tt is not None:self._takt_time()
        if self.model:
            std, mp_adj , mp = fetch_std_time(self.model, self.state.line, self.state.date)
            target = mp_adj if mp_adj is not None else std
            if target is not None:
                self.state.std_time = target
                self.state.ul_time = target * UL_FACTOR
                print("std = ",mp_adj)
    def _force_refresh_db(self) -> None:
        self.last_db_refresh = dt.datetime.min

    def _takt_time(self):
        now = dt.datetime.now()
        if self.ltk is not None:
            if self.ltk == self.tt:
                return
            print("tt : ",self.tt, "Type : ",type(self.tt))
            print("ltk : ",self.ltk, "Type : ",type(self.ltk))
            self.diff_seconds = (self.tt - self.ltk).total_seconds()
            append_csv("TT", now, float("{:.1f}".format(self.diff_seconds)), csv_filename)
            self.ltk = self.tt
            with open(csv_filename, "r", newline="") as f:
                r = csv.DictReader(f)
                for row in r:
                    try:
                        ts = dt.datetime.fromisoformat(row["timestamp"])
                    except Exception:
                        continue
                    self.state.processes["TT"].add(ts, float("{:.1f}".format(self.diff_seconds)))
        else:
            self.ltk = self.tt

    # ---------- drawing ----------
    def _draw_header(self) -> None:
        self.ax_header.clear(); self.ax_header.axis('off')
        line = self.state.line
        model = getattr(self, "model", None)
        dstr = self.state.date.strftime('%d-%m-%Y')
        status = "On Production"
        leader = getattr(self, "leader", "")

        title = f"Prototype Idea {line} : Visualization"
        self.ax_header.text(
            0.5, 0.62, title,
            ha='center', va='center', fontsize=18, weight='bold',
            transform=self.ax_header.transAxes
        )
        info = f"Real Time Mode   |   Status: {status}   |   Leader: {leader}   |   Line: {line}   |   Date: {dstr}   |   Model: {model}"
        self.ax_header.text(
            0.5, 0.18, info,
            ha='center', va='center', fontsize=11,
            transform=self.ax_header.transAxes,
            bbox=dict(boxstyle='round', facecolor='#eaf6ff', edgecolor='none', pad=0.5)
        )

    
    def _draw_all_day(self) -> None:
        ax = self.ax_all_day
        ax.clear()
        self._series_cache.clear()
        self.line_handles.clear()
        self.point_handles.clear()
        self.legend_all_day = None

        ax.set_title(f"All-Day Cycle Time Tracking • {self.state.line} • {self.state.date:%d %b %Y}",pad=8, fontsize=12, weight='bold')

        # lines + points + cache
        for pid, series in self.state.processes.items():
            if pid == "TT" or not series.samples:
                continue
            t = [dt.datetime.combine(self.state.date, s.ts.time()) for s in series.samples]
            v = [s.ct for s in series.samples]
            self._series_cache[pid] = (t, v)

            vis = self.series_visible.get(pid, True)

            line, = ax.plot(t, v, '-', label=pid, lw=0.8, color=self.proc_colors[pid])
            line.set_visible(vis)
            self.line_handles[pid] = line

            pts = ax.scatter(t, v, s=9, picker=True, color=self.proc_colors[pid], alpha=0.65)
            pts.set_visible(vis)
            self.point_handles[pid] = pts

        if self.state.std_time is not None:
            ax.axhline(self.state.std_time, ls='--', color='black', lw=1.0, label='Target')

        # shift shading
        day = self.state.date
        now_limit = dt.datetime.now() if self.state.date == dt.date.today() else dt.datetime.combine(day, dt.time(23, 59, 59))
        for (st, et, lbl) in SHIFT_SEGMENTS:
            sdt = dt.datetime.combine(day, st)
            edt = dt.datetime.combine(day, et)
            if now_limit < sdt:
                continue
            clipped_edt = min(edt, now_limit)
            ax.axvspan(
                sdt, clipped_edt,
                facecolor='#ffb3b3' if lbl in ['BREAK', 'LUNCH'] else '#d5ffd5',
                alpha=0.25
            )
            mid = sdt + (clipped_edt - sdt) / 2
            ax.text(mdates.date2num(mid), 0.98, lbl, ha='center', va='top', transform=ax.get_xaxis_transform())

        # formatting
        ax.set_ylim(0, 200)
        ax.set_ylabel('Cycle Time (s)')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.grid(True, axis='both', alpha=0.25)
        ax.set_position([self.bbox_all_day.x0, self.bbox_all_day.y0 * 0.95, self.bbox_all_day.width, self.bbox_all_day.height])

        leg = ax.legend(fontsize=7, ncol=4, loc='upper left',bbox_to_anchor=(0.0, -0.18), frameon=False)
        self.legend_all_day = leg
        leg.set_zorder(10)
        self._leg_handles = list(getattr(leg, "legendHandles", []))
        self._leg_texts   = list(leg.get_texts())

        # (Optional) visual cue: dim entries for series currently hidden
        for h, t in zip(self._leg_handles, self._leg_texts):
            label = t.get_text()
            if label in self.series_visible:
                alpha = 1.0 if self.series_visible[label] else 0.25
                try:
                    h.set_alpha(alpha)
                except Exception:
                    pass
                t.set_alpha(alpha)

        # draw (or clear) current highlight
        self._apply_all_day_highlight()
    
    '''
    def _draw_all_day(self) -> None:
        ax = self.ax_all_day

        if not self._all_day_init:
            # 1) STATIC once: axes cosmetics
            ax.set_title(f"All Day Cycle Time Tracking of {self.state.line} on {self.state.date:%d %b %Y}")
            ax.set_ylim(0, 200)
            ax.set_ylabel('Cycle Time (s)')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.grid(True, which='both', axis='both', alpha=0.3)
            ax.set_position([self.bbox_all_day.x0, self.bbox_all_day.y0 * 0.95,
                             self.bbox_all_day.width, self.bbox_all_day.height])

            # 2) STATIC once: shift spans
            self._static_spans.clear()
            day = self.state.date
            now_limit = dt.datetime.now() if day == dt.date.today() else dt.datetime.combine(day, dt.time(23,59,59))
            for (st, et, lbl) in SHIFT_SEGMENTS:
                sdt = dt.datetime.combine(day, st)
                edt = dt.datetime.combine(day, et)
                if now_limit >= sdt:
                    clipped_edt = min(edt, now_limit)
                    span = ax.axvspan(
                        sdt, clipped_edt,
                        facecolor='#ffb3b3' if lbl in ['BREAK', 'LUNCH'] else '#d5ffd5',
                        alpha=0.25
                    )
                    self._static_spans.append(span)
                    mid = sdt + (clipped_edt - sdt)/2
                    ax.text(mdates.date2num(mid), 0.98, lbl, ha='center', va='top',
                            transform=ax.get_xaxis_transform())

            # 3) STATIC once: target line (keep handle & label)
            if self.state.std_time is not None:
                ax.axhline(self.state.std_time, ls='--', color='black', lw=1.0, label='Target')

            # 4) DYNAMIC containers (one time create)
            self._series_cache.clear()
            self.line_handles.clear()
            self.point_handles.clear()

            for pid, series in self.state.processes.items():
                if pid == "TT" or not series.samples:
                    continue
                t = [dt.datetime.combine(self.state.date, s.ts.time()) for s in series.samples]
                v = [s.ct for s in series.samples]
                self._series_cache[pid] = (t, v)

                vis = self.series_visible.get(pid, True)
                line, = ax.plot([], [], '-', label=pid, lw=0.8, color=self.proc_colors[pid], animated=False)
                line.set_visible(vis)
                self.line_handles[pid] = line

                sc = ax.scatter([], [], s=9, picker=True, color=self.proc_colors[pid], alpha=0.65)
                sc.set_visible(vis)
                self.point_handles[pid] = sc

            # 5) Legend once (make both handle & text pickable)
            leg = ax.legend(ncol=1, fontsize=6.5)
            self.legend_all_day = leg
            self._legend_artist_to_label = {}
            handles = getattr(leg, "legendHandles", [])
            texts   = leg.get_texts()
            for h, t in zip(handles, texts):
                label = t.get_text()
                if hasattr(h, "set_picker"): h.set_picker(8)
                t.set_picker(True)
                self._legend_artist_to_label[h] = label
                self._legend_artist_to_label[t] = label
            for t in texts[len(handles):]:
                t.set_picker(True)
                self._legend_artist_to_label[t] = t.get_text()

            self._all_day_init = True

        # === UPDATE PATH (fast) ===
        # refresh data only
        for pid, series in self.state.processes.items():
            if pid == "TT" or not series.samples:
                continue
            t = [dt.datetime.combine(self.state.date, s.ts.time()) for s in series.samples]
            v = [s.ct for s in series.samples]
            self._series_cache[pid] = (t, v)

            if pid in self.line_handles:
                self.line_handles[pid].set_data(mdates.date2num(t), v)
            if pid in self.point_handles:
                # set_offsets expects Nx2 array of floats in data coords
                import numpy as _np
                self.point_handles[pid].set_offsets(_np.column_stack([mdates.date2num(t), v]))

        # refresh the current highlight (it draws overlays)
        self._apply_all_day_highlight()
        '''
    def _on_click_legend(self, event):
        """Toggle a series when the user clicks a legend text/handle."""
        leg = self.legend_all_day
        if leg is None:
            return
        if event.button != 1:  # left click only
            return
        # We only act when clicking inside the all_day axes area (legend lives there)
        if event.inaxes is not self.ax_all_day:
            # Some backends set inaxes=None for legend area; still try
            pass

        renderer = self.fig.canvas.get_renderer()
        if renderer is None:
            # Force a draw once to get a renderer (rare)
            self.fig.canvas.draw()
            renderer = self.fig.canvas.get_renderer()

        # 1) Check legend texts first (usually easiest to hit)
        for i, t in enumerate(getattr(self, "_leg_texts", [])):
            try:
                bbox = t.get_window_extent(renderer=renderer).expanded(1.1, 1.4)  # a bit forgiving
            except Exception:
                continue
            if bbox.contains(event.x, event.y):
                label = t.get_text()
                self._toggle_series_from_legend(label, i)
                return

        # 2) If not a text, try legend handles (lines/patches)
        for i, h in enumerate(getattr(self, "_leg_handles", [])):
            try:
                bbox = h.get_window_extent(renderer=renderer).expanded(1.2, 1.6)
            except Exception:
                # Some proxy artists don’t have a bbox; fall back to contains if available
                if hasattr(h, "contains") and h.contains(event)[0]:
                    # map handle index -> corresponding text label when possible
                    label = self._leg_texts[i].get_text() if i < len(self._leg_texts) else None
                    if label:
                        self._toggle_series_from_legend(label, i)
                    return
                continue
            if bbox.contains(event.x, event.y):
                label = self._leg_texts[i].get_text() if i < len(self._leg_texts) else None
                if label:
                    self._toggle_series_from_legend(label, i)
                return

    def _toggle_series_from_legend(self, label: str, idx: int):
        # Skip reference lines
        if label == "Target":
            return
        if label not in self.line_handles:
            return

        pid = label
        new_vis = not self.series_visible.get(pid, True)
        self.series_visible[pid] = new_vis

        # Toggle artists
        self.line_handles[pid].set_visible(new_vis)
        if pid in self.point_handles:
            self.point_handles[pid].set_visible(new_vis)

        # Clear highlight if we hide the highlighted series
        if not new_vis and self._hl_sel and self._hl_sel.get("pid") == pid:
            self._hl_sel = None
            self._apply_all_day_highlight()

        # Dim both the handle and the text for that legend entry (if present)
        if hasattr(self, "_leg_handles") and idx < len(self._leg_handles):
            try:
                self._leg_handles[idx].set_alpha(1.0 if new_vis else 0.25)
            except Exception:
                pass
        if hasattr(self, "_leg_texts") and idx < len(self._leg_texts):
            self._leg_texts[idx].set_alpha(1.0 if new_vis else 0.25)
        
        self._draw_rt_bar()           
        self.fig.canvas.draw_idle()

    def _apply_all_day_highlight(self):
        # remove old
        while self._hl_artists:
            a = self._hl_artists.pop()
            try: a.remove()
            except Exception: pass

        if not self._sel_time:
            return

        # vertical line at selected time
        vline = self.ax_all_day.axvline(self._sel_time, color='k', lw=2.0, alpha=0.7, zorder=8)
        self._hl_artists.append(vline)

        # optional: ring the last point before that time for every visible series
        for pid, (t, v) in self._series_cache.items():
            if not t:
                continue
            if not self.series_visible.get(pid, True):
                continue
            # find last index with t_i <= sel_time
            idx = None
            for i in range(len(t) - 1, -1, -1):
                if t[i] <= self._sel_time:
                    idx = i
                    break
            if idx is None:
                continue
            ring = self.ax_all_day.scatter(
                [t[idx]], [v[idx]], s=70,
                facecolors='none', edgecolors=self.proc_colors[pid], linewidths=1.6, zorder=9
            )
            self._hl_artists.append(ring)

    def _on_pick_all_day(self, event):
        artist = event.artist

        # --- Legend click? (handle or text) ---
        if hasattr(self, "_legend_artist_to_label") and artist in self._legend_artist_to_label:
            label = self._legend_artist_to_label[artist]

            # don't toggle the fixed 'Target' reference line
            if label == "Target":
                return

            if label in self.line_handles:
                pid = label
                new_vis = not self.series_visible.get(pid, True)
                self.series_visible[pid] = new_vis

                # toggle both the line and its scatter points
                self.line_handles[pid].set_visible(new_vis)
                if pid in self.point_handles:
                    self.point_handles[pid].set_visible(new_vis)

                # clear highlight if we're hiding the highlighted series
                if not new_vis and self._hl_sel and self._hl_sel.get("pid") == pid:
                    self._hl_sel = None
                    self._apply_all_day_highlight()

                # dim that legend entry (both handle + text)
                leg = self.legend_all_day
                if leg is not None:
                    for h, t in zip(getattr(leg, "legendHandles", []), leg.get_texts()):
                        if t.get_text() == label:
                            alpha = 1.0 if new_vis else 0.25
                            try:
                                if h is not None:
                                    h.set_alpha(alpha)
                            except Exception:
                                pass
                            t.set_alpha(alpha)
                            break

                self.fig.canvas.draw_idle()
            return  # handled
        # --- Point click? set / clear selected timestamp ---
        if isinstance(artist, PathCollection):
            pid = None
            for k, sc in self.point_handles.items():
                if sc is artist:
                    pid = k
                    break
            if pid is None or pid not in self._series_cache:
                return

            t, v = self._series_cache[pid]
            if not t:
                return

            inds = getattr(event, "ind", [])
            if not inds:
                return

            end_idx = int(inds[-1])
            clicked_time = t[end_idx]

            # ✅ toggle logic
            if self._sel_time and abs((self._sel_time - clicked_time).total_seconds()) < 1e-6:
                # same time clicked again → clear selection
                self._sel_time = None
            else:
                # new selection
                self._sel_time = clicked_time

            # re-draw all dependent plots
            self._apply_all_day_highlight()
            self._draw_rt_bar()
            self._draw_stats_tbl()
            self.fig.canvas.draw_idle()

    # -------- rt bar / tables / controls --------
    '''def _draw_rt_bar(self) -> None: 
        ax = self.ax_rt_bar
        ax2 = self.ax_fluc
        ax.clear()
        ax2.clear()
        ax.set_title(f"Real-Time Cycle Time Tracking of {self.state.line} on {self.state.date:%d %b %Y}")

        pids = self.state.proc_ids()
        x = np.arange(len(pids))
        avg_vals = [self.state.proc(pid).avgs() or 0 for pid in pids]
        fluct_vals = [self.state.proc(pid).fluct_pcts() or 0 for pid in pids]
        max_val = [self.state.proc(pid).ct_maxs() or 0 for pid in pids]
        min_val = [self.state.proc(pid).ct_min() or 0 for pid in pids]

        ax.bar(x, max_val, color='black', width=0.05, label='CT Max')
        bars = ax.bar(x, min_val, color=[self.proc_colors[p] for p in pids], width=0.6, label='CT Min')

        if self.state.std_time is not None:
            ax.axhline(self.state.std_time, linestyle='--', color='black', linewidth=1.0, label='Target')
        if self.state.ul_time is not None:
            ax.axhline(self.state.ul_time, linestyle=':', color='red', linewidth=1.0, label='Upper Limit')

        ax2.plot(x, fluct_vals, '-o', color='gray', label='% Fluctuation')
        ax2.plot(x, avg_vals, 'o', color='black', label='AVG CT')

        for xi, pid, rect, av, fl in zip(x, pids, bars, avg_vals, fluct_vals):
            is_ok = True
            if self.state.ul_time is not None and av > self.state.ul_time:
                is_ok = False
            ax.text(rect.get_x() + rect.get_width()/1.2, rect.get_height() + 0.5,
                    '✓' if is_ok else '✗',
                    ha='center', va='bottom',
                    color='green' if is_ok else 'red',
                    fontsize=14, weight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(pids)
        ax.set_xlabel('Process')
        ax.set_ylabel('Average CT (s)')
        ax.set_ylim(0, 200)
        ax2.set_ylim(0, 200)
        ax.grid(axis='y', alpha=0.3)

        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1, l1, loc='upper left', fontsize=8)
        ax2.legend(h2, l2, loc='upper right', fontsize=8)'''

        # Draw value labels on the bars if enableddef _draw_rt_bar(self) -> None:
    def _draw_rt_bar(self) -> None:
        ax, ax2 = self.ax_rt_bar, self.ax_fluc
        ax.clear(); ax2.clear()

        # remove old labels
        for L in (self._rt_bar_labels, self._rt_avg_plot_label, self._rt_max_plot_label):
            for t in L:
                try: t.remove()
                except: pass
        self._rt_bar_labels, self._rt_avg_plot_label, self._rt_max_plot_label = [], [], []

        ax.set_title(f"Real-Time Cycle Time Tracking of {self.state.line} on {self.state.date:%d %b %Y}")

        pids = self.state.proc_ids()
        x = np.arange(len(pids))

        # choose the base time: selected time or "now"
        base_time = self._sel_time
        if base_time is None:
            # fallback: latest timestamp we have (across all series)
            all_ts = [s.ts for p in self.state.processes.values() for s in p.samples]
            base_time = max(all_ts) if all_ts else dt.datetime.now()

        # compute stats from last 5 values BEFORE base_time
        avg_vals, fluct_vals, max_vals, min_vals = [], [], [], []
        for pid in pids:
            s = self.state.proc(pid)
            vals = s.values_before(base_time)
            if vals.size:
                last5 = vals[-5:]
                av = float(np.mean(last5))
                mn = float(np.min(last5))
                mx = float(np.max(last5))
                fl = ((mx - mn) / (av if av != 0 else 1.0)) * 100.0
            else:
                av = mn = mx = fl = 0.0
            avg_vals.append(av); min_vals.append(mn); max_vals.append(mx); fluct_vals.append(fl)

        max_bars = ax.bar(x, max_vals, color='black', width=0.05, label='CT Max', zorder=2)
        min_bars = ax.bar(x, min_vals, color=[self.proc_colors[p] for p in pids],
                        width=0.4, label='CT Min', zorder=1)

        # Target / UL
        if self.state.std_time is not None:
            ax.axhline(self.state.std_time, linestyle='--', color='black', linewidth=1.0, label='Target')
        if self.state.ul_time is not None:
            ax.axhline(self.state.ul_time, linestyle=':', color='red', linewidth=1.0, label='Upper Limit')

        # Fluctuation / Avg on twin axis
        ax2.plot(x, fluct_vals, '-o', color='gray', label='% Fluctuation')
        ax2.plot(x, avg_vals, 'o', color='black', label='AVG CT')
        # Optional numeric labels
        if self.show_rt_values:
            # Max labels (left axis)
            for rect, mx in zip(max_bars, max_vals):
                tmax = ax.text(rect.get_x() + rect.get_width() - 0.3, rect.get_height(),
                            f"{mx:.1f}", ha='center', va='bottom', fontsize=8)
                self._rt_max_plot_label.append(tmax)
            # Avg labels (right axis) near the points
            for xi, av in zip(x, avg_vals):
                tav = ax2.text(xi - 0.3, av, f"{av:.1f}", ha='center', va='bottom', fontsize=8)
                self._rt_avg_plot_label.append(tav)

        # ✓ / ✗ and (optionally) min labels
        for xi, pid, min_rect, av, mn in zip(x, pids, min_bars, avg_vals, min_vals):
            is_ok = True
            if self.state.ul_time is not None and av > self.state.ul_time:
                is_ok = False

            chk = ax.text(min_rect.get_x() + min_rect.get_width()/1.2,
                        min_rect.get_height() + 0.8,
                        '✓' if is_ok else '✗',
                        ha='center', va='bottom',
                        color=('green' if is_ok else 'red'),
                        fontsize=12, weight='bold')   # smaller font as requested
            self._rt_bar_labels.append(chk)

            if self.show_rt_values:
                tmin = ax.text(min_rect.get_x() + min_rect.get_width() - 0.4,
                            min_rect.get_height() + 0.8,
                            f"{mn:.1f}", ha='center', va='bottom', fontsize=8)
                self._rt_bar_labels.append(tmin)

        ax.set_xticks(x); ax.set_xticklabels(pids)
        ax.set_xlabel('Process'); ax.set_ylabel('CT (s)')
        ax.set_ylim(0, 200); ax2.set_ylim(0, 200)
        ax.grid(axis='y', alpha=0.3)

        # Legends
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1, l1, loc='upper left', fontsize=8, frameon=False)
        ax2.legend(h2, l2, loc='upper right', fontsize=8, frameon=False)

    def _ensure_rtbar_checkbox(self):
        if self._rtbar_ck is not None:
            return
        # Dock just above the RT bar, right-aligned within its column
        bbox = self.ax_rt_bar.get_position(self.fig)
        w, h = 0.06, 0.025
        left = bbox.x1 - w
        bottom = bbox.y1 + 0.006
        self._rtbar_ck_ax = self.fig.add_axes([left, bottom, w, h])
        self._rtbar_ck_ax.set_zorder(10)
        labels = ["Show Val"]
        states = [self.show_rt_values]
        self._rtbar_ck = CheckButtons(self._rtbar_ck_ax, labels, states)
        self._rtbar_ck.on_clicked(self._on_rtbar_checkbox)
        for label in self._rtbar_ck.labels:
            label.set_fontsize(7)

    def _on_rtbar_checkbox(self, label):
        # Toggle state
        self.show_rt_values = not self.show_rt_values
        # Redraw the RT bar with/without labels
        self._draw_rt_bar()
        self.fig.canvas.draw_idle()

    def _draw_prod_tbl(self) -> None:
        ax = self.ax_prod_tbl
        ax.clear()
        ax.axis('off')
        ax.set_title('Production Data', pad=2, fontsize=10, weight='bold')

        s = self.state.proc(pid="TT")
        avg_ct=_fmt(s.avg());ct_min= _fmt(s.ct_min());ct_max= _fmt(s.ct_max());fluct_pct= _fmt(s.fluct_pct(), pct=True)
        rows = []
        if getattr(self, "prod_data", None):
            for prd, lid, tno, plan, status, sdt, edt, cbid in self.prod_data:
                rows.append([
                    sdt, edt, getattr(self, "model", None), status, plan, getattr(self, "actual", None),
                    _fmt(self.state.std_time), ct_min, ct_max, avg_ct, fluct_pct
                ])
        col_labels = ['Start','End/Est End','Model','Status','Plan','Actual','TT','CTMin','CTMax','AVG CT','%Fluct']
        tbl = ax.table(cellText=rows, colLabels=col_labels, loc='center', cellLoc='center',bbox=[0.0, 0.0, 1.0, 1.0])
        tbl.auto_set_font_size(False)
        tbl.auto_set_column_width(range(len(col_labels)))
        tbl.set_fontsize(7)

    '''def _draw_stats_tbl(self) -> None:
        ax = self.ax_stats_tbl
        ax.clear()
        ax.axis('off')
        ax.set_title('Statistic Data', pad=15)
        pids = self.state.proc_ids()

        def row_for(pid: str):
            s = self.state.proc(pid)
            return [_fmt(s.avgs()), _fmt(s.ct_mins()), _fmt(s.ct_maxs()), _fmt(s.fluct_pcts(), pct=True)]

        matrix = [row_for(pid) for pid in pids]
        col_labels = ['AVG CT', 'CT Min', 'CT Max', '%Fluctuation']
        tbl = ax.table(
            cellText=list(map(list, zip(*matrix))),
            rowLabels=col_labels, colLabels=pids,
            cellLoc='center', loc='center',
            bbox=[0.15, -1, 0.8, 2.2]
        )
        tbl.auto_set_font_size(True)
        tbl.auto_set_column_width(range(len(pids)))
        tbl.set_fontsize(8)'''
    def _draw_stats_tbl(self) -> None:
        ax = self.ax_stats_tbl
        ax.clear(); ax.axis('off')
        ax.set_title('Latest 5 CT values before selected time (newest at top)', pad=4, fontsize=11, weight='bold')

        pids = [pid for pid in self.state.proc_ids() if pid != "TT"]

        # base time: selected or latest available
        base_time = self._sel_time
        if base_time is None:
            all_ts = [s.ts for p in self.state.processes.values() for s in p.samples]
            base_time = max(all_ts) if all_ts else dt.datetime.now()

        # Build 5 rows (newest first) from values BEFORE base_time
        rows_to_show = 5
        matrix = []
        for row_idx in range(rows_to_show):
            row = []
            for pid in pids:
                s = self.state.proc(pid)
                last5 = s.last_n_before(base_time, 5)
                if len(last5) > row_idx:
                    val = last5[-1 - row_idx].ct  # newest at index -1
                    row.append(_fmt(float(val)))
                else:
                    row.append('')
            matrix.append(row)

        row_labels = ['#1', '#2', '#3', '#4', '#5']  # #1 is newest
        tbl = ax.table(
            cellText=matrix, rowLabels=row_labels, colLabels=pids,
            cellLoc='center', loc='center', bbox=[0.0, 0.0, 1.0, 1.0]
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)

    def _draw_button(self) -> None:
        ax = self.ax_button
        ax.clear(); ax.axis('off')
        # Use the full tiny axis we reserved
        bbox = ax.get_position()
        btn_ax = self.fig.add_axes([bbox.x0, bbox.y0, bbox.width, bbox.height])
        self.pause_button = Button(btn_ax, 'Pause', color='#efefef', hovercolor='#d7ebff')
        self.pause_button.on_clicked(self._toggle_pause)

        ax2 = self.ax_step_btn
        ax2.clear(); ax2.axis('off')
        bbox2 = ax2.get_position()
        st_ax = self.fig.add_axes([bbox2.x0, bbox2.y0, bbox2.width, bbox2.height])
        self.step_button = Button(st_ax, 'StepTracking', color='#efefef', hovercolor='#d7ebff')
        self.step_button.on_clicked(lambda _e: self.open_steptracking_window())

    def _toggle_pause(self, event):
        self.pause = not self.pause
        self.pause_button.label.set_text('Resume' if self.pause else 'Pause')

    def _draw_drop_date(self) -> None:
        # Try Dropdown; if not available, build fallback once
        if not self._ensure_dropdown():
            self._ensure_date_fallback()

    def _list_available_dates(self):
        try:
            files = sorted(DATA_DIR.glob(f"data_{lines}_*.csv"))
            out = []
            for f in files:
                iso = f.stem.split("_")[-1]
                try:
                    dt.date.fromisoformat(iso)
                    out.append(iso)
                except Exception:
                    pass
            return sorted(set(out))
        except Exception as e:
            log.warning("List dates failed: %s", e)
            return []

    def _ensure_dropdown(self):
        if not Dropdown:
            return False  # use fallback

        if hasattr(self, "date_dropdown"):
            self._sync_dropdown_options()
            return True

        # Clear and hide the existing drop_date axis background
        host = self.ax_drop_date
        host.clear()
        host.axis("off")

        # Use the host's bbox in figure coords
        bbox = host.get_position(self.fig)

        # Make the dropdown occupy the top-right portion of ax_drop_date
        DD_W, DD_H = 0.9 * bbox.width, 0.8 * bbox.height  # relative to ax size
        left = bbox.x0 + (bbox.width - DD_W)
        bottom = bbox.y0 + (bbox.height - DD_H)

        ax_dd = self.fig.add_axes([left, bottom, DD_W, DD_H])
        ax_dd.set_zorder(10)
        ax_dd.set_facecolor("none")

        options = self._list_available_dates() or [self.state.date.isoformat()]
        cur = self.state.date.isoformat()
        if cur not in options:
            options = [cur] + options

        self._ax_date_dropdown = ax_dd
        self.date_dropdown = Dropdown(ax_dd, label="Date", options=options, value=cur)
        self.date_dropdown.on_select(self._on_date_select)
        return True

    def _sync_dropdown_options(self):
        cur = self.state.date.isoformat()
        options = self._list_available_dates() or [cur]
        if cur not in options:
            options = [cur] + options

        ax = self._ax_date_dropdown
        ax.cla()
        ax.set_zorder(10)
        ax.set_facecolor("none")
        self.date_dropdown = Dropdown(ax, label="Date", options=options, value=cur)
        self.date_dropdown.on_select(self._on_date_select)

    def _on_date_select(self, label: str):
        try:
            new_date = dt.date.fromisoformat(label)
        except Exception:
            log.warning("Bad date from dropdown: %s", label)
            return
        self._load_date(new_date)

    # ---- Fallback path: TextBox + Prev/Next ----
    def _ensure_date_fallback(self):
        if hasattr(self, "date_text"):
            return

        host = self.ax_drop_date
        host.clear()
        host.axis("off")

        bbox = host.get_position(self.fig)

        BTN_W, BTN_H = 0.02, 0.04
        TXT_W, TXT_H = 0.08, 0.04
        GAP = 0.006

        # Translate host's top-right corner to figure coords
        right = bbox.x1
        top = bbox.y1
        bottom = top - TXT_H

        # Arrange inside ax_drop_date's bbox
        next_left = right - BTN_W
        text_left = next_left - GAP - TXT_W
        prev_left = text_left - GAP - BTN_W

        #prev_ax = self.fig.add_axes([prev_left, bottom, BTN_W, BTN_H])
        text_ax = self.fig.add_axes([text_left, bottom, TXT_W, TXT_H])
        #next_ax = self.fig.add_axes([next_left, bottom, BTN_W, BTN_H])

        #for a in (prev_ax, text_ax, next_ax):
        #    a.set_zorder(10)
        #    a.set_facecolor("none")

        #self.date_prev_btn = Button(prev_ax, "◀")
        self.date_text     = TextBox(text_ax, "Date ", initial=self.state.date.isoformat())
        #self.date_next_btn = Button(next_ax, "▶")

        self.date_text.on_submit(self._on_date_submit)
        #self.date_prev_btn.on_clicked(lambda _e: self._shift_date(-1))
        #self.date_next_btn.on_clicked(lambda _e: self._shift_date(+1))

    def _on_date_submit(self, text: str):
        try:
            new_date = dt.date.fromisoformat(text)   # parse string → date
        except Exception:
            log.warning("Invalid date: %s", text)
            self.date_text.set_val(self.state.date.isoformat())
            return
        self._load_date(new_date)


    def _shift_date(self, delta: int):
        dates = self._list_available_dates()
        if not dates:
            return
        cur = self.state.date.isoformat()
        if cur not in dates:
            dates = sorted(set(dates + [cur]))
        i = max(0, min(len(dates)-1, dates.index(cur) + delta))
        self._load_date(dt.date.fromisoformat(dates[i]))

    # ---- Shared loader ----
    def _load_date(self, new_date: dt.date):
        self.data_dir = f"data_{lines}_{new_date}.csv"
        for p in self.state.processes.values():
            p.samples.clear()
        load_csv(self.state, self.data_dir)
        self._sel_time = None
        if hasattr(self, "date_dropdown"):
            self.date_dropdown.set_val(new_date)
        if hasattr(self, "date_text"):
            self.date_text.set_val(new_date)

    def open_steptracking_window(self):
        """Open a second page/window that shows StepTracking skipped records (video + table)."""
        # Reuse if already open
        if hasattr(self, "_fig_step") and plt.fignum_exists(self._fig_step.number):
            self._refresh_steptracking_table()  # also keeps current video
            plt.figure(self._fig_step.number)
            return

        self._fig_step = plt.figure(figsize=(14, 7), dpi=110)
        try:
            self._fig_step.canvas.manager.set_window_title("StepTracking • Skipped")
        except Exception:
            pass

        gs = self._fig_step.add_gridspec(
            12, 12, left=0.05, right=0.98, top=0.95, bottom=0.08, wspace=0.35, hspace=0.3
        )
        self._ax_step_title = self._fig_step.add_subplot(gs[0:2, 0:12]); self._ax_step_title.axis('off')
        self._ax_step_video = self._fig_step.add_subplot(gs[2:12, 0:7])   # LEFT: video
        self._ax_step_table = self._fig_step.add_subplot(gs[2:12, 7:12])  # RIGHT: table
        self._ax_step_title.text(0.5, 0.5, "StepTracking — Skipped (NAS)", ha='center', va='center',
                                fontsize=15, weight='bold', transform=self._ax_step_title.transAxes)

        # Video state
        self._step_cap = None
        self._step_im = None
        self._step_video_timer = self._fig_step.canvas.new_timer(interval=33)  # ~30fps
        self._step_video_timer.add_callback(self._update_step_video)

        # Table interaction map
        self._step_cell_to_index = {}
        self._step_rows: List[Tuple[dt.datetime, str, str]] = []  # (ts, filename_only, full_path)

        # Click handler for table
        self._fig_step.canvas.mpl_connect('button_press_event', self._on_step_table_click)

        # First draw
        self._refresh_steptracking_table()
        self._fig_step.canvas.draw_idle()


    def _refresh_steptracking_table(self):
        """Fetch skipped records for self.state.date and render the clickable table."""
        ax = self._ax_step_table
        ax.clear(); ax.axis('off')

        rows = fetch_steptracking_skipped_for_date(self.state.date)
        self._step_rows = rows  # store

        if not rows:
            ax.text(0.5, 0.5, "No skipped records found for this date.", ha='center', va='center', fontsize=11)
            self._fig_step.canvas.draw_idle()
            return

        # Build table data: show only filename + human timestamp (no path)
        table_rows = []
        for i, (ts, filename_only, _full) in enumerate(rows, start=1):
            table_rows.append([i, ts.strftime("%Y-%m-%d %I:%M:%S %p"), filename_only])

        col_labels = ["#", "time_error", "filename"]
        tbl = ax.table(cellText=table_rows, colLabels=col_labels, loc='center',
                    cellLoc='center', colLoc='center', bbox=[0.0, 0.0, 1.0, 1.0])
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.auto_set_column_width(col=list(range(len(col_labels))))

        # Make the index & filename look clickable; map cells -> row index
        self._step_cell_to_index.clear()
        cells = tbl.get_celld()
        n_rows = len(table_rows)
        n_cols = len(col_labels)
        for r in range(1, n_rows+1):  # data rows (r=0 is header)
            for c in range(n_cols):
                cell = cells[(r, c)]
                cell.set_edgecolor("#dddddd")
                if c in (0, 2):  # index & filename clickable
                    txt = cell.get_text()
                    txt.set_color("#1a73e8")
                    try: txt.set_underline(True)
                    except Exception: pass
                self._step_cell_to_index[cell] = r-1  # map to zero-based data row

        # Auto-select/play the first row if no video loaded yet
        if self._step_cap is None and self._step_rows:
            _ts0, _fn0, path0 = self._step_rows[0]
            self._load_step_video(path0, display_title=_fn0)

        self._fig_step.canvas.draw_idle()

    def _on_step_table_click(self, event):
        """Play selected video when a table cell is clicked."""
        if event.button != 1 or event.inaxes is None:
            return
        if not hasattr(self, "_fig_step") or event.canvas != self._fig_step.canvas:
            return

        renderer = self._fig_step.canvas.get_renderer()
        for cell, idx in list(self._step_cell_to_index.items()):
            try:
                if cell.get_window_extent(renderer).contains(event.x, event.y):
                    # Load the clicked video
                    _ts, filename_only, full_path = self._step_rows[idx]
                    self._load_step_video(full_path, display_title=filename_only)
                    break
            except Exception:
                continue

    def _load_step_video(self, path: str, display_title: Optional[str] = None):
        """Open a video from UNC path and start/continue playback."""
        # Close previous
        try:
            if self._step_cap is not None:
                self._step_cap.release()
        except Exception:
            pass

        self._ax_step_video.clear()
        self._ax_step_video.axis('off')

        # Show what we're trying to load
        title = display_title or os.path.basename(path)
        self._ax_step_video.set_title(title, fontsize=10, pad=6)

        if not os.path.exists(path):
            self._ax_step_video.text(0.5, 0.5, f"File not found:\n{title}", ha='center', va='center', fontsize=11)
            self._step_cap = None
            self._step_video_timer.stop()
            self._fig_step.canvas.draw_idle()
            return

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            self._ax_step_video.text(0.5, 0.5, f"Cannot open video:\n{title}", ha='center', va='center', fontsize=11)
            self._step_cap = None
            self._step_video_timer.stop()
            self._fig_step.canvas.draw_idle()
            return

        # Read first frame
        ok, frame = cap.read()
        if not ok or frame is None:
            self._ax_step_video.text(0.5, 0.5, f"Empty or unreadable video:\n{title}", ha='center', va='center', fontsize=11)
            cap.release()
            self._step_cap = None
            self._step_video_timer.stop()
            self._fig_step.canvas.draw_idle()
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._step_im = self._ax_step_video.imshow(frame_rgb, animated=True)
        self._step_cap = cap
        self._step_video_timer.start()
        self._fig_step.canvas.draw_idle()

    def _update_step_video(self):
        if self._step_cap is None:
            return
        ok, frame = self._step_cap.read()
        if not ok or frame is None:
            try:
                self._step_cap.release()
            except Exception:
                pass
            self._step_cap = None
            self._step_video_timer.stop()
            return
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self._step_im is None:
            self._step_im = self._ax_step_video.imshow(frame_rgb, animated=True)
        else:
            self._step_im.set_data(frame_rgb)
        self._fig_step.canvas.draw_idle()

    def start_live_camera(self):
        self.live_im = None
        self.anim = FuncAnimation(self.fig, self._update_live_camera, interval=33)

    def _update_live_camera(self, _):
        if not hasattr(self, 'ax_live') or not self.livestream:
            return

        frame = self.livestream.get_frame()
        if frame is None:
            return
        
        bbox = self.ax_live.get_window_extent()
        width_px = int(bbox.width)
        height_px = int(bbox.height)

        # Resize camera frame to match subplot
        frame_resized = cv2.resize(frame, (width_px, height_px))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        if self.live_im is None:
            self.ax_live.axis("off")
            self.live_im = self.ax_live.imshow(frame_rgb,animated=True)
        else:
            self.ax_live.axis("off")
            self.live_im.set_data(frame_rgb)
        return [self.live_im]


    def _stats_for_pid(self, pid: str, override_end_idx: Optional[int] = None, k: int = 5):
        """
        Return stats over a window of up to k samples.
        If override_end_idx is given, use the all-day cache window that ends at that index.
        Otherwise, use the last k values from ProcessSeries.values.
        """
        if override_end_idx is not None and pid in self._series_cache:
            t, v = self._series_cache[pid]
            if not v:
                return {"avg": 0.0, "min": 0.0, "max": 0.0, "fluct": 0.0}
            end = int(override_end_idx)
            start = max(0, end - (k - 1))
            window = np.array(v[start:end + 1], dtype=float)
        else:
            vals = self.state.proc(pid).values
            if vals.size == 0:
                return {"avg": 0.0, "min": 0.0, "max": 0.0, "fluct": 0.0}
            window = vals[-k:] if vals.size > k else vals

        if window.size == 0:
            return {"avg": 0.0, "min": 0.0, "max": 0.0, "fluct": 0.0}

        mn = float(np.min(window))
        mx = float(np.max(window))
        av = float(np.mean(window))
        fluctu = ((mx - mn) / (av if av != 0 else 1.0)) * 100.0
        return {"avg": av, "min": mn, "max": mx, "fluct": fluctu}

# ----------------------------------------------------------------------------
# DATA I/O HELPERS (CSV ROTATION)
# ----------------------------------------------------------------------------

def load_csv_incremental(state: DashboardState, path: str, dash) -> None:
    if not os.path.exists(path): return
    # reset if file rolled over (new day)
    if dash._last_loaded_file != path:
        dash._last_loaded_file = path
        dash._csv_pos = 0
        for proc in state.processes.values():
            proc.samples.clear()

    with open(path, "r", newline="") as f:
        if dash._csv_pos == 0:
            # skip header
            header = f.readline()
            dash._csv_pos = f.tell()

        f.seek(dash._csv_pos)
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != 3:
                continue
            ts = dt.datetime.fromisoformat(parts[0].split("timestamp:")[-1]) if "timestamp" in parts[0] else dt.datetime.fromisoformat(parts[0])
            pid = parts[1].split("process_id:")[-1] if "process_id" in parts[1] else parts[1]
            ct  = float(parts[2].split("cycle_time:")[-1]) if "cycle_time" in parts[2] else float(parts[2])
            if pid in state.processes:
                state.processes[pid].add(ts, ct)
        dash._csv_pos = f.tell()

CSV_HEADER = ["timestamp", "process_id", "cycle_time"]

def csv_path(data_dir: Path, line: str, d: dt.date) -> Path:
    return data_dir / f"data_{line}_{d:%Y%m%d}.csv"

def load_csv(state: DashboardState, data_dir) -> None:
    if not os.path.exists(data_dir):
        return
    
    for proc in state.processes.values():
        proc.samples.clear()

    with open(data_dir,"r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                ts = dt.datetime.fromisoformat(row["timestamp"])
                pid = row["process_id"]
                ct = float(row["cycle_time"])
            except Exception:
                continue
            if pid in state.processes:
                state.processes[pid].add(ts, ct)

def append_csv(pid: str, ts: dt.datetime, ct: float,data_dir) -> None:
    header_needed = not os.path.exists(data_dir)
    with open(data_dir,"a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if header_needed:
            w.writeheader()
        if ct > std_time*0.6:
            w.writerow({"timestamp": ts.isoformat(), "process_id": pid, "cycle_time": ct})

# ----------------------------------------------------------------------------
# UTIL
# ----------------------------------------------------------------------------

def _fmt(val: Optional[float], pct: bool = False) -> str:
    if val is None:
        return ''
    if pct:
        return f"{val:.1f}%"
    if isinstance(val, float):
        if abs(val - round(val)) < 1e-6:
            return f"{val:.0f}"
        return f"{val:.1f}"
    return str(val)


# ----------------------------------------------------------------------------
# MAIN APP LOOP
# ----------------------------------------------------------------------------

def run_app(args: argparse.Namespace) -> None:
    line = args.line
    n_proc = args.nproc
    data_dir = csv_filename
    video_dir = Path(args.video_dir)
    video_dir.mkdir(exist_ok=True, parents=True)
    if args.date is None:
        today = dt.date.today()
    else:
        try:
            today = dt.datetime.strptime(args.date, '%Y-%m-%d').date()
        except ValueError:
            log.error("Bad --date format; expected YYYY-MM-DD; using today")
            today = dt.date.today()

    state = DashboardState(line=line, n_proc=n_proc, date=today, std_time=args.std_time)
    if state.std_time is not None:
        state.ul_time = state.std_time * UL_FACTOR

    # load historical csv for the day (if any)
    load_csv(state, data_dir)

    # Acquire frame placeholder if user passed --live-frame
    if args.live_frame:
        state.live_frame_path = Path(args.live_frame)

    # Enable interactive mode for matplotlib
    plt.ion()
    live_stream = LiveStream(src=LIVE_STREAM_SRC)
    live_stream.start()
    dash = DY08Dashboard(state, data_dir, video_dir,live_stream)
    dash.start_live_camera()

    # inbound queue for both MQTT + simulator
    in_q: queue.Queue = queue.Queue()

    # MQTT live ingest (unless disabled or simulate)
    mqtt_ing = None
    if  not args.no_mqtt:
        mqtt_cfg = MQTTConfig()
        mqtt_ing = MQTTIngest(mqtt_cfg, lines, in_q)
        mqtt_ing.start()

    # Matplotlib timer for refresh & simulation ------------------------------
    def on_timer(event):
        global current_date
        # Check for new day and create new CSV if needed
        today = dt.date.today()
        if today != current_date:
            current_date = today
            csv_filename = f"data_{line}_{current_date}.csv"
            # Create the new day's CSV file with header immediately
            path = data_dir / csv_filename
            if not path.exists():
                with path.open('w', newline='') as f:
                    fieldnames = ['timestamp', 'process_id', 'cycle_time']
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
        # redraw
        dash.refresh(conn_str)

    # Use a positive interval for timer
    timer_interval = max(REFRESH_UI_SEC, 1) * 1000
    timer = dash.fig.canvas.new_timer(interval=timer_interval)
    timer.add_callback(on_timer, None)
    timer.start()
    log.info("Starting dashboard... (press space to pause, q to quit)")

    try:
        from IPython import get_ipython  # type: ignore
        in_ipy = get_ipython() is not None
    except Exception:
        in_ipy = False

    if in_ipy:
        plt.show(block=False)   # don't block notebook
        return dash, timer
    else:
        plt.show(block=True)
        return dash, timer
    # cleanup

# ----------------------------------------------------------------------------
# CLI ARGUMENT PARSING
# ----------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='DY-08 Production Dashboard Prototype')
    p.add_argument('--line', default=LINE_DEFAULT, help='Line identifier (e.g., DY-08)')
    p.add_argument('--nproc', type=int, default=mp, help='Number of processes')
    p.add_argument('--std-time', type=float, default=None, help='Target cycle time (sec) override')
    p.add_argument('--date', default=None, help='Date YYYY-MM-DD (defaults today)')
    p.add_argument('--data-dir', default=None, help='Directory for per-day CSV logs')
    p.add_argument('--video-dir', default=str(VIDEO_DIR_DEFAULT), help='Directory of process video clips')
    p.add_argument('--live-frame', default=None, help='Path to still image to show in Live panel')
    p.add_argument('--no-db', action='store_true', help='Disable DB lookups even if env vars set')
    p.add_argument('--no-mqtt', action='store_true', help='Disable MQTT ingest')
    return p

# ----------------------------------------------------------------------------
# ENTRY POINT
# ----------------------------------------------------------------------------

if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    run_app(args)
