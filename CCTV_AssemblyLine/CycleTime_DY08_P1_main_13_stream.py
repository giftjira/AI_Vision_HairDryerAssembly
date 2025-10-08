import cv2
import numpy as np
import time
from datetime import datetime
import tkinter as tk
import threading
from tkinter import *
from tkinter import ttk
from PIL import Image, ImageDraw
import csv
import os
from tkinter import messagebox
import paho.mqtt.client as mqtt
import tflite_runtime.interpreter as tflite
import subprocess
import re

# =========================
# Globals & Constants
# =========================
app = None

# Model constants
EMPTY = 0
NOT_EMPTY = 1
OTHER = 2

# Network and MQTT constants
interface_name = "wlp1s0"  # Replace with your actual WiFi interface name
interfaces = ['wlp1s0']
firewall_port = 1883

# Program parameters
frame_rate = 30.0
WiFi_Reconect_step = 1000  # (unused currently, keep for future)
step_size = 5
resize_margin = 10
external_padx = 10
external_pady = 10
internal_padx = 10
internal_pady = 5

# Program state / defaults
File_Name = "N/A"
Device_Addr = "N/A"
Host_Addr = "N/A"
WiFi_SSID = "N/A"
WiFi_USERNAME = "N/A"
WiFi_PWD = "N/A"
Camera_Addr = 5
LstStat_Cam = "Good"
LstStat_PwdOff = "Bad"
LstStat_WiFi = "Bad"
LstStat_MQTT = "Bad"
LstStat_ExitProg = "Bad"
LstStat_LogicLoop = "Bad"

# Additional parameters
resize_shape = (100, 100, 3)   # Unused (kept for ref)
default_spot = [10, 10, 50, 50]
first_spot_position = [0, 0, 0, 0]
diff_threshold = 0.1  # Unused (kept for ref)

# Colors (BGR)
background_color = (0, 0, 0)
empty_color = (0, 0, 255)      # Red for EMPTY
not_empty_color = (0, 255, 0)  # Green for NOT_EMPTY
other_color = (255, 0, 0)      # Blue for OTHER
selected_color = (0, 255, 255) # Yellow for selected spot
text_color = (0, 0, 0)         # Black text
adjust_text_color = (50, 50, 255)

# Runtime vars
Model_ID = "N/A"
std_time = "N/A"
cycle_time = 0.0
step = 30  # default; can be overridden by const_data CSV

# >>> TFLite threads: use env var or half of cores (min 1)
NUM_THREADS = int(os.getenv("TFLITE_THREADS", max(1, (os.cpu_count() or 4)//2)))

# Thread stop handle
stop_event = threading.Event()

# =========================
# Safe filename parsing
# =========================
try:
    base = os.path.basename(__file__)
    parts = base.split('_')
    if len(parts) >= 3:
        line_id = parts[1]
        # parts[2] may include extension; strip it
        filename = f"{parts[1]}_{os.path.splitext(parts[2])[0]}"
    else:
        line_id = "line0"
        filename = os.path.splitext(base)[0]
except Exception:
    line_id = "line0"
    filename = "unknown"
File_Name = filename

# =========================
# MQTT callbacks
# =========================
def on_connect(client, userdata, flags, rc):
    global LstStat_MQTT
    if rc == 0:
        client.on_subscribe = on_subscribe
        client.subscribe(f"{line_id}/Model_ID")
        LstStat_MQTT = "Good"
        update_ConstData()
        reload_models_for_new_model_id()
    else:
        print(f"Failed to connect, return code {rc}")
        LstStat_MQTT = "Bad"
        update_ConstData()

def on_subscribe(client, userdata, mid, granted_qos):
    print("Subscribed successfully. mid =", mid, "granted_qos =", granted_qos)

def on_message(client, userdata, msg):
    global Model_ID, std_time
    topic = msg.topic
    payload = msg.payload.decode()
    if topic.endswith("Model_ID"):
        # Expecting "MODELID_STD" format; be defensive
        parts = str(payload).split('_', 1)
        Model_ID = parts[0] if len(parts) > 0 else "N/A"
        std_time = parts[1] if len(parts) > 1 else "N/A"
        print(f"Received new Model_ID: {Model_ID} | std_time: {std_time}")
        update_ConstData()
        reload_models_for_new_model_id()

def reload_models_for_new_model_id():
    global MainSub, filename
    if not MainSub:
        return
    for S in [sublist[0] for sublist in MainSub]:
        new_model_path = f"dataset/models/{Model_ID}/model_{filename}_S{S}.tflite"
        if not os.path.exists(new_model_path):
            print(f"[WARNING] {new_model_path} does not exist. Using default model.")
            new_model_path = "dataset/model_default.tflite"
        try:
            new_interpreter = tflite.Interpreter(model_path=new_model_path, num_threads=NUM_THREADS)
            new_interpreter.allocate_tensors()
            globals()[f'interpreter{S}'] = new_interpreter
            globals()[f'model_file{S}'] = new_model_path
            print(f"Loaded new model for S={S}: {new_model_path}")
        except Exception as ex:
            print(f"Failed to load new model for S={S}: {ex}")
            fallback = "dataset/model_default.tflite"
            new_interpreter = tflite.Interpreter(model_path=fallback, num_threads=NUM_THREADS)
            new_interpreter.allocate_tensors()
            globals()[f'interpreter{S}'] = new_interpreter
            globals()[f'model_file{S}'] = fallback

def on_publish(client, userdata, mid):
    print("Message published.")

def on_log(client, userdata, level, buf):
    print("LOG:", level, buf)

# =========================
# Mount NAS (best-effort)
# =========================
def mount_nas():
    try:
        subprocess.run(['sudo', 'mkdir','-p','/mnt/nas'], check=False)
        subprocess.run(
            ['sudo', 'mount', '-t', 'cifs',
             '//158.118.87.150/pmfth-nas/CameraCapture-AI-Vision', '/mnt/nas',
             '-o', 'username=09PCB,password=Edpisc#01,uid=1000,gid=1000,rw,vers=3.0'],
            check=True
        )
    except subprocess.CalledProcessError:
        print("[WARN] NAS mount failed (continuing).")

mount_nas()

# =========================
# Network helpers
# =========================
def connect_to_wifi(ssid, username, password, iface):
    global LstStat_WiFi
    try:
        subprocess.run(['sudo','nmcli', 'connection', 'delete', 'id', ssid], check=False)
        subprocess.run([
            'sudo','nmcli', 'connection', 'add',
            'type', 'wifi',
            'con-name', ssid,
            'ifname', iface,
            'ssid', ssid,
            'wifi-sec.key-mgmt', 'wpa-eap',
            '802-1x.eap', 'peap',
            '802-1x.phase2-auth', 'mschapv2',
            '802-1x.identity', username,
            '802-1x.password', password
        ], check=True)
        subprocess.run(['sudo','nmcli', 'connection', 'up', ssid], check=True)
        LstStat_WiFi = "Good"
        print("Connected to WiFi.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to connect to WiFi. Error: {e}")
        LstStat_WiFi = "Bad"

def check_and_reconnect():
    global LstStat_WiFi
    while not stop_event.is_set():
        try:
            connection_status = subprocess.run(
                ['sudo','nmcli', '-t', '-f', 'DEVICE,STATE', 'dev'],
                capture_output=True, text=True, check=False
            )
            connected = any('connected' in line for line in connection_status.stdout.splitlines())
            if not connected or LstStat_WiFi == "Bad":
                LstStat_WiFi = "Bad"
                connect_to_wifi(WiFi_SSID, WiFi_USERNAME, WiFi_PWD, interface_name)
            else:
                LstStat_WiFi = "Good"
        except Exception:
            LstStat_WiFi = "Bad"
        time.sleep(10)  # prevent CPU spin

def update_ConstData():
    global filename, File_Name, Device_Addr, Host_Addr, WiFi_SSID, WiFi_USERNAME, WiFi_PWD
    global Camera_Addr, LstStat_Cam, LstStat_PwdOff, LstStat_WiFi, LstStat_MQTT
    global LstStat_ExitProg, LstStat_LogicLoop, Model_ID, step

    ConstData_filename = f"dataset/const_data_{filename}.csv"
    Const_data = [[
        File_Name, Device_Addr, Host_Addr, WiFi_SSID, WiFi_USERNAME, WiFi_PWD,
        Camera_Addr, LstStat_Cam, LstStat_PwdOff, LstStat_WiFi, LstStat_MQTT,
        LstStat_ExitProg, LstStat_LogicLoop, Model_ID, step
    ]]

    os.makedirs(os.path.dirname(ConstData_filename), exist_ok=True)
    with open(ConstData_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            'File_Name','Device_Addr','Host_Addr','WiFi_SSID','WiFi_USERNAME','WiFi_PWD',
            'Camera_Addr','LstStat_Cam','LstStat_PwdOff','LstStat_WiFi','LstStat_MQTT',
            'LstStat_ExitProg','LstStat_LogicLoop','Model_ID','Step'
        ])
        writer.writerows(Const_data)

def get_camera_index():
    result = subprocess.run(['v4l2-ctl', '--list-devices'], stdout=subprocess.PIPE)
    output = result.stdout.decode('utf-8', errors='ignore')
    matches = re.findall(r'/dev/video(\d+)', output)
    indices = [int(m) for m in matches]
    if 5 in indices: return 5
    if 6 in indices: return 6
    return indices[0] if indices else 0

def get_ip_address(interfaces_list):
    for interface in interfaces_list:
        try:
            result = subprocess.run(['ip', 'addr', 'show', interface],
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode != 0:
                print(f"Error running ip addr show on {interface}: {result.stderr}")
                continue
            match = re.search(r'inet (\d+\.\d+\.\d+\.\d+)', result.stdout)
            if match:
                return match.group(1)
        except Exception as e:
            print(f"An error occurred with interface {interface}: {e}")
            continue
    print("No IP address found for any interface")
    return None

# =========================
# Vision helpers
# =========================
def get_spots_boxes(connected_components):
    (totalLabels, label_ids, values, centroid) = connected_components
    slots = []
    for i in range(1, totalLabels):
        x1 = int(values[i, cv2.CC_STAT_LEFT])
        y1 = int(values[i, cv2.CC_STAT_TOP])
        w  = int(values[i, cv2.CC_STAT_WIDTH])
        h  = int(values[i, cv2.CC_STAT_HEIGHT])
        slots.append([x1, y1, w, h])
    return slots

def calc_diff(im1, im2):
    return np.abs(np.mean(im1) - np.mean(im2))

def empty_or_not(spot_bgr, interpreter, input_details, output_details):
    img_rgb = cv2.cvtColor(spot_bgr, cv2.COLOR_BGR2RGB)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    IMG_HEIGHT, IMG_WIDTH = input_shape[1], input_shape[2]
    img_resized = cv2.resize(img_rgb, (IMG_WIDTH, IMG_HEIGHT))
    img_input = np.expand_dims(img_resized, axis=0).astype(np.uint8)
    interpreter.set_tensor(input_details[0]['index'], img_input)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(output_data, axis=1)[0]
    return predicted_class

def GetStructureFromCSV(file_path):
    main_box_counts = {}
    if not os.path.exists(file_path):
        return []
    with open(file_path, mode='r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader, None)
        for row in csv_reader:
            if not row: continue
            main_box = int(row[0])
            main_box_counts[main_box] = main_box_counts.get(main_box, 0) + 1
    return [[main_box, count] for main_box, count in main_box_counts.items()]

def GetConstDataFromCSV(file_path):
    global File_Name, Device_Addr, Host_Addr, WiFi_SSID, WiFi_USERNAME, WiFi_PWD
    global Camera_Addr, LstStat_Cam, LstStat_PwdOff, LstStat_WiFi, LstStat_MQTT
    global LstStat_ExitProg, LstStat_LogicLoop, Model_ID, step

    if not os.path.exists(file_path):
        return [[
            File_Name, Device_Addr, Host_Addr, WiFi_SSID, WiFi_USERNAME, WiFi_PWD,
            Camera_Addr, LstStat_Cam, LstStat_PwdOff, LstStat_WiFi, LstStat_MQTT,
            LstStat_ExitProg, LstStat_LogicLoop, Model_ID, step
        ]]

    with open(file_path, mode='r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader, None)
        for row in csv_reader:
            if not row: continue
            File_Name = row[0]
            Device_Addr = row[1]
            Host_Addr = row[2]
            WiFi_SSID = row[3]
            WiFi_USERNAME = row[4]
            WiFi_PWD = row[5]
            Camera_Addr = row[6]
            LstStat_Cam = row[7]
            LstStat_PwdOff = row[8]
            LstStat_WiFi = row[9]
            LstStat_MQTT = row[10]
            LstStat_ExitProg = row[11]
            LstStat_LogicLoop = row[12]
            Model_ID = row[13]
            step = int(row[14]) if len(row) > 14 and row[14].isdigit() else step

    return [[
        File_Name, Device_Addr, Host_Addr, WiFi_SSID, WiFi_USERNAME, WiFi_PWD,
        Camera_Addr, LstStat_Cam, LstStat_PwdOff, LstStat_WiFi, LstStat_MQTT,
        LstStat_ExitProg, LstStat_LogicLoop, Model_ID, step
    ]]

def create_config_data(spots, main_box, model_file):
    config_data = []
    for i, spot in enumerate(spots):
        x, y, w, h = spot
        sub_box = i + 1
        config_data.append([main_box, sub_box, model_file, x, y, w, h])
    return config_data

def create_box_mask(boxes, image_size, background_color, output_path):
    image = Image.new("RGB", image_size, background_color)
    draw = ImageDraw.Draw(image)
    for box in boxes:
        x, y, w, h = box
        draw.rectangle([x, y, x + w, y + h], fill=(255, 255, 255))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image.save(output_path)

def boxes_overlap(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    return (x1 < x2 + w2 and x1 + w1 > x2 and y1 < y2 + h2 and y1 + h1 > y2)

def any_boxes_overlap(boxes):
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes_overlap(boxes[i], boxes[j]):
                return True
    return False

def resolve_other(prev_state: int, curr_state: int) -> int:
    if curr_state == OTHER:
        return prev_state if prev_state in (EMPTY, NOT_EMPTY) else EMPTY
    return curr_state

# =========================
# Pre-load structure & consts BEFORE MQTT
# =========================
MainSub = GetStructureFromCSV(f"dataset/config_data_{filename}.csv")
Const_data = GetConstDataFromCSV(f"dataset/const_data_{filename}.csv")

# Camera & device info
Camera_Addr = get_camera_index()
print("Found camera address:", Camera_Addr)

Device_Addr = get_ip_address(interfaces)
if Device_Addr:
    print(f"IP address found: {Device_Addr}")
else:
    print("Failed to get IP address for any interface")

update_ConstData()

# Bring up Wi-Fi before MQTT
connect_to_wifi(WiFi_SSID, WiFi_USERNAME, WiFi_PWD, interface_name)
wifi_thread = threading.Thread(target=check_and_reconnect, daemon=True)
wifi_thread.start()

# =========================
# Build spot/config structures and load models
# =========================
Config_data = []
for S in [sublist[0] for sublist in MainSub]:
    model_path = f"dataset/model_{filename}_S{S}.tflite"
    if os.path.exists(model_path):
        try:
            interpreter = tflite.Interpreter(model_path=model_path, num_threads=NUM_THREADS)
            interpreter.allocate_tensors()
        except Exception as e:
            print(f"Failed to load model for S={S}: {e}")
            continue
    else:
        model_path = 'dataset/model_default.tflite'
        interpreter = tflite.Interpreter(model_path=model_path, num_threads=NUM_THREADS)
        interpreter.allocate_tensors()

    mask = f"dataset/mask_img_{filename}_S{S}.png"
    model_file = os.path.basename(model_path)
    mask_img = cv2.imread(mask, 0) if os.path.exists(mask) else None
    if mask_img is None:
        # if mask missing, create empty mask file sized later (after camera size is known)
        os.makedirs(os.path.dirname(mask), exist_ok=True)
        Image.new("RGB", (640, 480), background_color).save(mask)  # temp
        mask_img = cv2.imread(mask, 0)

    spots = get_spots_boxes(cv2.connectedComponentsWithStats(mask_img, 4, cv2.CV_32S))
    spots_status = [None for _ in spots]
    spots_raw_status = [None for _ in spots]
    previous_spots_status = [None for _ in spots]
    original_spots = spots.copy()

    Config_data.extend(create_config_data(spots, S, model_file))

    globals()[f'interpreter{S}'] = interpreter
    globals()[f'mask{S}'] = mask
    globals()[f'model_file{S}'] = model_file
    globals()[f'mask{S}_img'] = mask_img
    globals()[f'spots{S}'] = spots
    globals()[f'spots_status{S}'] = spots_status
    globals()[f'spots_raw_status{S}'] = spots_raw_status
    globals()[f'previous_spots_status{S}'] = previous_spots_status
    globals()[f'original_spots{S}'] = original_spots
    globals()[f'selected_object_index{S}'] = None

# =========================
# MQTT setup (after Host_Addr known & Wi-Fi up)
# =========================
client = mqtt.Client(protocol=mqtt.MQTTv311)
client.on_connect = on_connect
client.on_message = on_message
client.on_publish = on_publish
client.on_log = on_log

try:
    client.connect(Host_Addr, firewall_port, 60)
    client.loop_start()
except Exception as e:
    print("MQTT connection failed:", e)
    LstStat_MQTT = "Bad"
    update_ConstData()

# =========================
# Video & Streaming
# =========================
cap = cv2.VideoCapture(Camera_Addr)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS,30)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
ok = cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
image_size = (frame_width, frame_height)
ffmpeg_cmd = [
    "ffmpeg", "-y",
    "-f", "rawvideo", "-pix_fmt", "bgr24",
    "-s", f"{int(cap.get(3))}x{int(cap.get(4))}",
    "-r", "25",
    "-i", "-",
    "-an",
    "-c:v", "libx264",
    "-preset", "ultrafast",
    "-tune", "zerolatency",
    "-g", "25",
    "-x264-params", "keyint=25:min-keyint=25:scenecut=0:bframes=0:rc-lookahead=0",
    "-f", "rtsp",
    "-rtsp_transport", "udp",      # <-- tell FFmpeg to use RTP/UDP
    "rtsp://10.84.171.19:8554/mystream",
]

process = None  # spawn later only after first successful frame

mask_height = frame_height
mask_width = frame_width

start_time = time.time()
t_E = t_S = t_prev_E = None
e_detected = s_detected = False
previous_frame = None
assembly_time = "N/A"

# =========================
# Tkinter GUI
# =========================
class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()
        self.show_idx = False
        self.is_adjust = False
        self.is_recording = False
        self.video_writer = None
        self.adjust_window = None
        self.config_window = None
        self.is_image_capturing = False
        self.image_capture_folder = None

    def create_widgets(self):
        self.master.geometry("320x280+1+1")
        self.container = tk.Frame(self.master)
        self.container.pack(fill="both", expand=True)
        self.frames = {}
        for F in (StartPage, RecordPage, AdjustPage):
            page_name = F.__name__
            frame = F(parent=self.container, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")
        self.show_frame("StartPage")

    def show_frame(self, page_name):
        if page_name == "StartPage":
            self.is_adjust = False
            for S in [sublist[0] for sublist in MainSub]:
                globals()[f'spots{S}'] = globals()[f'original_spots{S}'].copy()
        if page_name == "AdjustPage":
            self.is_adjust = True
        frame = self.frames[page_name]
        frame.tkraise()

    def show_index(self):
        self.show_idx = not self.show_idx

    def start_recording(self):
        global filename
        if not self.is_recording:
            self.is_recording = True
            current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            video_filename = f"/mnt/nas/cam_video_{filename}_{current_datetime}.mp4"
            codec = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(video_filename, codec, frame_rate, image_size)
            print(f"Recording started: {video_filename}")

    def stop_recording(self):
        if self.is_recording:
            self.is_recording = False
            if self.video_writer is not None:
                self.video_writer.release()
                print("Recording stopped.")
            messagebox.showinfo("Recording Stopped", "Recording has stopped.")
        else:
            print("No active recording to stop.")

    def start_image_capture(self):
        global filename
        if not self.is_image_capturing:
            self.is_image_capturing = True
            current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.image_capture_folder = f"dataset/image_capture_{filename}_{current_datetime}"
            os.makedirs(self.image_capture_folder, exist_ok=True)
            print(f"Image capture started: {self.image_capture_folder}")
        else:
            print("Image capture is already running.")

    def stop_image_capture(self):
        if self.is_image_capturing:
            self.is_image_capturing = False
            self.image_capture_folder = None
            print("Image capture stopped.")
            messagebox.showinfo("Image Capture Stopped", "Image capture has stopped.")
        else:
            print("No active image capture to stop.")

    def save_adjustments(self):
        global Config_data, filename
        if any(any_boxes_overlap(globals()[f'spots{sublist[0]}']) for sublist in MainSub):
            overlapping_sublist = next(
                sublist for sublist in MainSub if any_boxes_overlap(globals()[f'spots{sublist[0]}']))
            messagebox.showinfo(
                "Message",
                f"Sub Boxes are overlapping on Main Box No. {overlapping_sublist[0]}\n"
                f"Please check the overlapping again before saving."
            )
        else:
            Config_data.clear()
            for S in [sublist[0] for sublist in MainSub]:
                Config_data.extend(create_config_data(globals()[f'spots{S}'], S, globals()[f'model_file{S}']))
            for S in [sublist[0] for sublist in MainSub]:
                create_box_mask(globals()[f'spots{S}'], image_size, background_color, globals()[f'mask{S}'])
            for S in [sublist[0] for sublist in MainSub]:
                globals()[f'original_spots{S}'] = globals()[f'spots{S}'].copy()

            csv_filename = f"dataset/config_data_{filename}.csv"
            os.makedirs(os.path.dirname(csv_filename), exist_ok=True)
            if os.path.exists(csv_filename):
                try:
                    os.remove(csv_filename)
                except PermissionError as e:
                    print(f"Error: {e}")

            with open(csv_filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Main Box', 'Sub Box', 'Model File', 'X', 'Y', 'W', 'H'])
                writer.writerows(Config_data)

            print(f"Config_data saved to {csv_filename}")
            self.is_adjust = True
            self.update_treeview()

    def reset_adjustments(self):
        self.is_adjust = True
        for S in [sublist[0] for sublist in MainSub]:
            globals()[f'spots{S}'] = globals()[f'original_spots{S}'].copy()

    def add_box(self):
        global filename
        adjust_page = self.frames["AdjustPage"]
        selected_main = adjust_page.selected_main.get()

        if selected_main == "New":
            new_main_index = max([i[0] for i in MainSub], default=0) + 1
            MainSub.append([new_main_index, 0])

            globals()[f'model{new_main_index}'] = None  # optional, not used
            globals()[f'spots{new_main_index}'] = []
            globals()[f'spots_status{new_main_index}'] = []
            globals()[f'original_spots{new_main_index}'] = []
            globals()[f'model_file{new_main_index}'] = 'model_default.tflite'
            globals()[f'mask{new_main_index}'] = f"dataset/mask_img_{filename}_S{new_main_index}.png"

            # Create an example image for the mask
            os.makedirs(os.path.dirname(globals()[f'mask{new_main_index}']), exist_ok=True)
            Image.new("RGB", image_size, background_color).save(globals()[f'mask{new_main_index}'])

            choices = [str(sublist[0]) for sublist in MainSub] + ["New"]
            adjust_page.selected_main.set(choices[0])

            menu = adjust_page.dropdown["menu"]
            menu.delete(0, "end")
            for choice in choices:
                menu.add_command(label=choice, command=tk._setit(adjust_page.selected_main, choice))

            spots_list = globals()[f'spots{new_main_index}']
            selected_main = new_main_index
        else:
            selected_main = int(selected_main)
            spots_list = globals().get(f'spots{selected_main}', [])

        if spots_list:
            first_spot = spots_list[0]
            new_spot = [0, 0, first_spot[2], first_spot[3]]
        else:
            new_spot = default_spot

        spots_list.append(new_spot)
        globals()[f'spots{selected_main}'] = spots_list
        globals()[f'spots_status{selected_main}'].append(None)

        self.save_adjustments()

    def delete_box(self):
        global MainSub, filename
        adjust_page = self.frames["AdjustPage"]
        selected_main = adjust_page.selected_main.get()
        if selected_main == "New":
            print("Cannot delete sub box from 'New' selection.")
            return

        selected_main_int = int(selected_main)
        selected_index = globals().get(f'selected_object_index{selected_main_int}', None)
        if selected_index is not None:
            spots_list = globals()[f'spots{selected_main_int}']
            if 0 <= selected_index < len(spots_list):
                spots_list.pop(selected_index)
                globals()[f'spots_status{selected_main_int}'].pop(selected_index)
                globals()[f'previous_spots_status{selected_main_int}'].pop(selected_index)
                globals()[f'original_spots{selected_main_int}'].pop(selected_index)
                globals()[f'selected_object_index{selected_main_int}'] = None
                self.save_adjustments()
                print(f"Deleted spot at index {selected_index} from Main box {selected_main_int}")
                MainSub = GetStructureFromCSV(f"dataset/config_data_{filename}.csv")
                menu = adjust_page.dropdown["menu"]
                menu.delete(0, "end")
                choices = [str(sublist[0]) for sublist in MainSub] + ["New"]
                for choice in choices:
                    menu.add_command(label=choice, command=tk._setit(adjust_page.selected_main, choice))
        else:
            print("No box selected to delete.")

    def delete_main_box(self):
        global MainSub
        adjust_page = self.frames["AdjustPage"]
        selected_main = adjust_page.selected_main.get()
        if selected_main == "New":
            print("Cannot delete 'New'.")
            return
        selected_main_int = int(selected_main) - 1
        if 0 <= selected_main_int < len(MainSub):
            confirm = messagebox.askyesno("Confirm Deletion",
                                          f"The deletion can't be undone.\n"
                                          f"Are you sure to delete the Main Box No. {selected_main}?")
            if confirm:
                del MainSub[selected_main_int]
                self.save_adjustments()
                print(f"Deleted Main box {selected_main}")
                choices = [str(sublist[0]) for sublist in MainSub] + ["New"]
                adjust_page.selected_main.set(choices[0])
                menu = adjust_page.dropdown["menu"]
                menu.delete(0, "end")
                for choice in choices:
                    menu.add_command(label=choice, command=tk._setit(adjust_page.selected_main, choice))
        else:
            print(f"Index {selected_main_int} is out of range.")

    def update_treeview(self):
        if hasattr(self, 'my_table') and self.my_table:
            for item in self.my_table.get_children():
                self.my_table.delete(item)
            for i, entry in enumerate(Config_data):
                main_box, sub_box, model_file, x, y, w, h = entry
                self.my_table.insert(parent='', index='end', iid=f'entry_{i}', text='',
                                     values=(main_box, sub_box, model_file, x, y, w, h))

    def configuration(self):
        def create_table_columns(table, columns):
            table['columns'] = columns
            for col in columns:
                table.column(col, anchor=tk.CENTER,
                             width=50 if col not in ['main_box', 'sub_box', 'model_file']
                             else 70 if col in ['main_box','sub_box'] else 130)
                table.heading(col, text=col.replace('_', ' ').title(), anchor=tk.CENTER)
            table.column("#0", width=0, stretch=tk.NO)
            table.heading("#0", text="", anchor=tk.CENTER)

        self.config_window = tk.Toplevel(self.master)
        self.config_window.title("Configuration Data")
        self.config_window.geometry('490x200+1+220')
        self.config_window['bg'] = '#AC99F2'

        table_frame = tk.Frame(self.config_window)
        table_frame.pack()

        self.my_table = ttk.Treeview(table_frame)
        columns = ('main_box', 'sub_box', 'model_file', 'x', 'y', 'w', 'h')
        create_table_columns(self.my_table, columns)

        self.update_treeview()

        scrollbar = tk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.my_table.yview)
        self.my_table.configure(yscrollcommand=scrollbar.set)
        self.my_table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

class StartPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.create_widgets()

    def create_widgets(self):
        self.show_index_button = tk.Button(self, text="Display Index", command=self.controller.show_index,
                                           padx=internal_padx, pady=internal_pady)
        self.show_index_button.grid(row=0, column=0, padx=external_padx, pady=external_pady)

        self.record_button = tk.Button(self, text="Video Recorder",
                                       command=lambda: self.controller.show_frame("RecordPage"),
                                       padx=internal_padx, pady=internal_pady)
        self.record_button.grid(row=1, column=0, padx=external_padx, pady=external_pady)

        self.adjust_button = tk.Button(self, text="Box Editor Mode",
                                       command=lambda: self.controller.show_frame("AdjustPage"),
                                       padx=internal_padx, pady=internal_pady)
        self.adjust_button.grid(row=0, column=1, padx=external_padx, pady=external_pady)

        self.config_button = tk.Button(self, text="Configuration Data",
                                       command=self.controller.configuration,
                                       padx=internal_padx, pady=internal_pady)
        self.config_button.grid(row=1, column=1, padx=external_padx, pady=external_pady)

class AdjustPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.selected_main = tk.StringVar(self)
        self.create_widgets()

    def create_widgets(self):
        choices = [str(sublist[0]) for sublist in MainSub] + ["New"]
        self.selected_main.set(choices[0] if choices else "New")
        self.dropdown = tk.OptionMenu(self, self.selected_main, *choices)
        self.dropdown.grid(row=1, column=0, padx=external_padx, pady=external_pady)

        text_main_selection = tk.Label(self, text="[Select Main Box No.]\n  V\n  V", font=("Helvetica", 9, "bold"))
        text_main_selection.grid(row=0, column=0)

        add_button = tk.Button(self, text="(+)Add Main/Sub Box", command=self.controller.add_box,
                               padx=internal_padx, pady=internal_pady)
        add_button.grid(row=0, column=1, padx=external_padx, pady=external_pady)

        delete_button = tk.Button(self, text="(-)Delete Sub Box", command=self.controller.delete_box,
                                  padx=internal_padx, pady=internal_pady)
        delete_button.grid(row=2, column=1, padx=external_padx, pady=external_pady)

        delete_mainbutton = tk.Button(self, text="(-)Delete Main Box", command=self.controller.delete_main_box,
                                      padx=internal_padx, pady=internal_pady)
        delete_mainbutton.grid(row=1, column=1, padx=external_padx, pady=external_pady)

        back_button = tk.Button(self, text="<<Back", command=lambda: self.controller.show_frame("StartPage"),
                                padx=internal_padx, pady=internal_pady)
        back_button.grid(row=2, column=0, padx=external_padx, pady=external_pady)

        save_button = tk.Button(self, text="Save", command=self.controller.save_adjustments,
                                padx=internal_padx, pady=internal_pady)
        save_button.grid(row=0, column=2, padx=external_padx, pady=external_pady)

        reset_button = tk.Button(self, text="Reset", command=self.controller.reset_adjustments,
                                 padx=internal_padx, pady=internal_pady)
        reset_button.grid(row=1, column=2, padx=external_padx, pady=external_pady)

class RecordPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.create_widgets()

    def create_widgets(self):
        start_video_button = tk.Button(self, text="Start Video Recording",
                                       command=self.controller.start_recording,
                                       padx=internal_padx, pady=internal_pady)
        start_video_button.pack(padx=external_padx, pady=external_pady)

        stop_video_button = tk.Button(self, text="Stop Video Recording",
                                      command=self.controller.stop_recording,
                                      padx=internal_padx, pady=internal_pady)
        stop_video_button.pack(padx=external_padx, pady=external_pady)

        start_image_button = tk.Button(self, text="Start Image Capture",
                                       command=self.controller.start_image_capture,
                                       padx=internal_padx, pady=internal_pady)
        start_image_button.pack(padx=external_padx, pady=external_pady)

        stop_image_button = tk.Button(self, text="Stop Image Capture",
                                      command=self.controller.stop_image_capture,
                                      padx=internal_padx, pady=internal_pady)
        stop_image_button.pack(padx=external_padx, pady=external_pady)

        back_button = tk.Button(self, text="<<Back",
                                command=lambda: self.controller.show_frame("StartPage"),
                                padx=internal_padx, pady=internal_pady)
        back_button.pack(padx=external_padx, pady=external_pady)

def update_gui():
    global app
    root = tk.Tk()
    root.title("Control Panel")
    app = Application(master=root)
    app.mainloop()

gui_thread = threading.Thread(target=update_gui, daemon=True)
gui_thread.start()

# =========================
# Mouse & Display
# =========================
dragging = False
resizing = False
resizing_edge = None

def mouse_events(event, x, y, flags, param):
    global dragging, resizing, resizing_edge, frame

    window_name = f'Real-Time Monitor ({filename})'
    actual_width = frame.shape[1] if frame is not None else 1
    actual_height = frame.shape[0] if frame is not None else 1

    try:
        _, _, window_w, window_h = cv2.getWindowImageRect(window_name)
    except Exception:
        window_w, window_h = actual_width, actual_height

    scale_x = actual_width / window_w if window_w else 1.0
    scale_y = actual_height / window_h if window_h else 1.0
    x = int(x * scale_x)
    y = int(y * scale_y)

    if app is not None and app.is_adjust:
        for S in [sublist[0] for sublist in MainSub]:
            if event == cv2.EVENT_LBUTTONDOWN:
                for i, spot in enumerate(globals()[f'spots{S}']):
                    x1, y1, w, h = spot
                    left_edge = x1
                    right_edge = x1 + w
                    top_edge = y1
                    bottom_edge = y1 + h

                    if (left_edge - resize_margin <= x <= left_edge + resize_margin and top_edge <= y <= bottom_edge):
                        globals()[f'selected_object_index{S}'] = i
                        resizing = True
                        resizing_edge = 'left'
                        break
                    elif (right_edge - resize_margin <= x <= right_edge + resize_margin and top_edge <= y <= bottom_edge):
                        globals()[f'selected_object_index{S}'] = i
                        resizing = True
                        resizing_edge = 'right'
                        break
                    elif (top_edge - resize_margin <= y <= top_edge + resize_margin and left_edge <= x <= right_edge):
                        globals()[f'selected_object_index{S}'] = i
                        resizing = True
                        resizing_edge = 'top'
                        break
                    elif (bottom_edge - resize_margin <= y <= bottom_edge + resize_margin and left_edge <= x <= right_edge):
                        globals()[f'selected_object_index{S}'] = i
                        resizing = True
                        resizing_edge = 'bottom'
                        break

                if not resizing:
                    for i, spot in enumerate(globals()[f'spots{S}']):
                        x1, y1, w, h = spot
                        if x1 <= x <= x1 + w and y1 <= y <= y1 + h:
                            globals()[f'selected_object_index{S}'] = i if globals()[f'selected_object_index{S}'] != i else None
                            dragging = True
                            break
                    else:
                        globals()[f'selected_object_index{S}'] = None

            elif event == cv2.EVENT_MOUSEMOVE:
                if dragging and globals()[f'selected_object_index{S}'] is not None:
                    x1, y1, w, h = globals()[f'spots{S}'][globals()[f'selected_object_index{S}']]
                    globals()[f'spots{S}'][globals()[f'selected_object_index{S}']] = [x - w // 2, y - h // 2, w, h]
                elif resizing and globals()[f'selected_object_index{S}'] is not None:
                    x1, y1, w, h = globals()[f'spots{S}'][globals()[f'selected_object_index{S}']]
                    if resizing_edge == 'left':
                        new_w = (x1 + w) - x
                        if new_w > 10:
                            globals()[f'spots{S}'][globals()[f'selected_object_index{S}']] = [x, y1, new_w, h]
                    elif resizing_edge == 'right':
                        new_w = x - x1
                        if new_w > 10:
                            globals()[f'spots{S}'][globals()[f'selected_object_index{S}']][2] = new_w
                    elif resizing_edge == 'top':
                        new_h = (y1 + h) - y
                        if new_h > 10:
                            globals()[f'spots{S}'][globals()[f'selected_object_index{S}']] = [x1, y, w, new_h]
                    elif resizing_edge == 'bottom':
                        new_h = y - y1
                        if new_h > 10:
                            globals()[f'spots{S}'][globals()[f'selected_object_index{S}']][3] = new_h

            elif event == cv2.EVENT_LBUTTONUP:
                dragging = False
                resizing = False
                resizing_edge = None

cv2.namedWindow(f'Real-Time Monitor ({filename})', cv2.WINDOW_NORMAL)
cv2.setWindowProperty(f'Real-Time Monitor ({filename})', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.setMouseCallback(f'Real-Time Monitor ({filename})', mouse_events)

def show_model_id_on_frame(frame):
    global Model_ID
    text_pos = (10, 30)
    cv2.putText(frame, f"Model: {Model_ID}", text_pos,
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

# =========================
# Main loop
# =========================
frame_nmr = 0
ret = True
cycle_counter = 1
cycle_time_start = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        LstStat_Cam = "Bad"
        update_ConstData()
        break
    else:
        LstStat_Cam = "Good"

    # Lazy-spawn ffmpeg once we know camera is good
    if process is None:
        try:
            process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
        except Exception as e:
            print(f"[WARN] Failed to start ffmpeg: {e}")
            process = None

    copy_frame = frame.copy()
    adjust_text_position = (frame.shape[1] - 170, 65)
    time_text_position = (frame.shape[1] - 360, 32)

    if frame_nmr % step == 0:
        # (Re)load interpreters & cache IO details (kept from your original logic;
        # if you want *even more* FPS, remove this reload and keep interpreters persistent)
        for S in [sublist[0] for sublist in MainSub]:
            model_path = f"dataset/model_{filename}_S{S}.tflite"
            if os.path.exists(model_path):
                try:
                    interpreter = tflite.Interpreter(model_path=model_path, num_threads=NUM_THREADS)
                    interpreter.allocate_tensors()
                except Exception as e:
                    print(f"Failed to load model for S={S}: {e}")
                    continue
            else:
                model_path = 'dataset/model_default.tflite'
                interpreter = tflite.Interpreter(model_path=model_path, num_threads=NUM_THREADS)
                interpreter.allocate_tensors()
            globals()[f'interpreter{S}'] = interpreter

        for S in [sublist[0] for sublist in MainSub]:
            interpreter = globals().get(f'interpreter{S}', None)
            if interpreter is None:
                continue
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            globals()[f'input_details{S}'] = input_details
            globals()[f'output_details{S}'] = output_details

            spots = globals()[f'spots{S}']
            for spot_indx, spot in enumerate(spots):
                x1, y1, w, h = spot
                # bounds safety
                x1 = max(0, min(x1, frame.shape[1]-1))
                y1 = max(0, min(y1, frame.shape[0]-1))
                w = max(1, min(w, frame.shape[1]-x1))
                h = max(1, min(h, frame.shape[0]-y1))

                spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
                raw_state = empty_or_not(
                    spot_crop,
                    globals()[f'interpreter{S}'],
                    globals()[f'input_details{S}'],
                    globals()[f'output_details{S}']
                )
                globals()[f'spots_raw_status{S}'][spot_indx] = raw_state
                prev_state = globals()[f'previous_spots_status{S}'][spot_indx]
                resolved_state = resolve_other(prev_state, raw_state)
                globals()[f'spots_status{S}'][spot_indx] = resolved_state

        # -------- Custom logic example (E stage) --------
        global spots_status1
        global previous_spots_status1
        if 'spots_status1' in globals() and 'previous_spots_status1' in globals():
            not_empty_count1 = sum(1 for s in spots_status1 if s == NOT_EMPTY)
            previous_not_empty_count1 = sum(1 for s in previous_spots_status1 if s == NOT_EMPTY)

            if not e_detected and not_empty_count1 > previous_not_empty_count1:
                t_E = time.time() - start_time
                e_detected = True
                LstStat_LogicLoop = "Good"
                print("E stage detected at: {:.1f} seconds".format(t_E))
                if t_prev_E is not None:
                    cycle_time = (t_E - t_prev_E)
                    e_detected = False
                    print("Cycle time: {}".format(cycle_time))
                else:
                    e_detected = False
                t_prev_E = t_E

        previous_frame = frame.copy()
        for S in [sublist[0] for sublist in MainSub]:
            spots_status = globals()[f'spots_status{S}']
            globals()[f'previous_spots_status{S}'] = spots_status.copy()

    display_frame = frame.copy()
    show_model_id_on_frame(display_frame)

    for S in [sublist[0] for sublist in MainSub]:
        spots = globals()[f'spots{S}']
        spots_status_raw = globals()[f'spots_raw_status{S}']
        for spot_indx, spot in enumerate(spots):
            spot_status = spots_status_raw[spot_indx]
            x1, y1, w, h = spots[spot_indx]
            color = empty_color if spot_status == EMPTY else not_empty_color if spot_status == NOT_EMPTY else other_color
            display_frame = cv2.rectangle(display_frame, (x1, y1), (x1 + w, y1 + h), color, 2)
            if app and app.show_idx:
                texts = f"{S}/{spot_indx + 1}"
                textDim = cv2.getTextSize(texts, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.putText(display_frame, texts,
                            (int(x1 + w // 2 - (textDim[0][0]) / 2),
                             int(y1 + h // 2 + (textDim[0][1]) / 2)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
            if globals()[f'selected_object_index{S}'] == spot_indx:
                cv2.rectangle(display_frame, (x1 - 2, y1 - 2), (x1 + w + 2, y1 + h + 2), selected_color, 3)

    current_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    if app and app.is_adjust:
        cv2.putText(display_frame, "Box Editor Mode", adjust_text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    adjust_text_color, 2, cv2.LINE_AA)
    cv2.putText(display_frame, current_time, time_text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2, cv2.LINE_AA)

    # Push to ffmpeg if running
    if process and process.stdin:
        try:
            process.stdin.write(display_frame.tobytes())
        except BrokenPipeError:
            print("[WARN] ffmpeg pipe broken; disabling stream.")
            process = None
        except Exception as e:
            print(f"[WARN] ffmpeg write error: {e}")
            process = None

    # Show resized preview window
    resized = cv2.resize(display_frame, (960, 720))
    cv2.imshow(f'Real-Time Monitor ({filename})', resized)

    # Save video if recording
    if app is not None and app.is_recording and app.video_writer is not None:
        app.video_writer.write(copy_frame)

    # Image capture logic
    if app is not None and app.is_image_capturing:
        N = 150
        if frame_nmr % N == 0:
            for S in [sublist[0] for sublist in MainSub]:
                spots = globals()[f'spots{S}']
                for spot_indx, spot in enumerate(spots):
                    x1, y1, w, h = spot
                    spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
                    spot_folder = os.path.join(app.image_capture_folder, f"Spot_{S}_{spot_indx + 1}")
                    os.makedirs(spot_folder, exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    image_filename = os.path.join(spot_folder, f"frame_{frame_nmr}_{timestamp}.jpg")
                    cv2.imwrite(image_filename, spot_crop)
                    print(f"Saved image: {image_filename}")

    key = cv2.waitKey(25) & 0xFF
    if key == ord('q'):
        LstStat_ExitProg = "Good"
        update_ConstData()
        break

    frame_nmr += 1
    LstStat_PwdOff = "Bad"

# =========================
# Shutdown & Cleanup
# =========================
LstStat_PwdOff = "Good"
update_ConstData()

try:
    client.loop_stop()
    client.disconnect()
except Exception:
    pass

stop_event.set()
try:
    wifi_thread.join(timeout=2)
except Exception:
    pass

try:
    cap.release()
except Exception:
    pass

if process:
    try:
        if process.stdin:
            process.stdin.close()
        process.wait(timeout=2)
    except Exception:
        pass

try:
    cv2.destroyAllWindows()
except Exception:
    pass

try:
    if app and app.master:
        app.master.after(0, app.master.quit)
except Exception:
    pass
