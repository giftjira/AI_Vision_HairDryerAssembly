import cv2
import numpy as np
#import pickle
#from skimage.transform import resize
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
import tensorflow as tf
#from tensorflow.keras.models import load_model
import tflite_runtime.interpreter as tflite

import subprocess
import re

app = None

## Define model constants 
EMPTY = 0
NOT_EMPTY = 1
OTHER = 2

## Network and MQTT constants
interface_name = "wlp1s0"  # Replace with your actual WiFi interface name
#interfaces = ['wlp1s0', 'wlan0']
interfaces = ['wlp1s0']
firewall_port = 1883

## Program parameters
frame_rate = 20.0
#step = 30
WiFi_Reconect_step = 1000 # will fine tune later
step_size = 5
resize_margin = 10
external_padx = 10
external_pady = 10
internal_padx = 10
internal_pady = 5

## Program Constants
File_Name = "N/A" #<= will came from code title
Device_Addr = "N/A"
#Host_Addr = "172.20.10.4"
Host_Addr = "10.84.171.108"
WiFi_SSID = "N/A"
WiFi_USERNAME = "N/A"
WiFi_PWD = "N/A"
Camera_Addr = 5
LstStat_Cam = "Good"
LstStat_PwdOff = "Bad"
LstStat_WiFi = "Bad" # TBC
LstStat_MQTT = "Bad"
LstStat_ExitProg = "Bad"
LstStat_LogicLoop = "Bad"

## Additional parameters
resize_shape = (100, 100, 3)  # Shape for resizing the spot
default_spot = [10, 10, 50, 50]  # Default dimensions for a new spot
first_spot_position = [0, 0, 0, 0]  # Initial position for the first spot
diff_threshold = 0.1  # Threshold for detecting significant differences

## Color codes
background_color = (0, 0, 0)
empty_color = (0, 0, 255)  # Red color for EMPTY
not_empty_color = (0, 255, 0)  # Green color for NOT_EMPTY
other_color = (255, 0, 0)  # Blue color for OTHER
selected_color = (0, 255, 255)  # Yellow color for selected spot
text_color = (0, 0, 0)  # White color for text
adjust_text_color = (50, 50, 255)

# **Add a global variable for Model_ID** (initially "N/A")
Model_ID = "N/A"

# Initialize cycle_time to a numeric value so we can publish it safely
cycle_time = 0.0

## MQTT setup section

def on_connect(client, userdata, flags, rc):
    global LstStat_MQTT, filename, cycle_time, Device_Addr, Host_Addr, WiFi_SSID, WiFi_USERNAME, WiFi_PWD, Camera_Addr, LstStat_Cam, LstStat_PwdOff, LstStat_WiFi, LstStat_MQTT, LstStat_ExitProg, LstStat_LogicLoop, Model_ID
    if rc == 0:
        #print("Connected successfully.")
        # After connecting, publish a message to a topic
        client.publish(f"{filename}/cycle_time", "{:.1f}".format(cycle_time))
        client.publish(f"{filename}/File_Name", File_Name)
        client.publish(f"{filename}/Device_Addr", Device_Addr)
        client.publish(f"{filename}/Host_Addr", Host_Addr)
        client.publish(f"{filename}/WiFi_SSID", WiFi_SSID)
        client.publish(f"{filename}/WiFi_USERNAME", WiFi_USERNAME)
        client.publish(f"{filename}/WiFi_PWD", WiFi_PWD)
        client.publish(f"{filename}/Camera_Addr", Camera_Addr)
        client.publish(f"{filename}/LstStat_Cam", LstStat_Cam)
        client.publish(f"{filename}/LstStat_PwdOff", LstStat_PwdOff)
        client.publish(f"{filename}/LstStat_WiFi", LstStat_WiFi)
        client.publish(f"{filename}/LstStat_MQTT", LstStat_MQTT)
        client.publish(f"{filename}/LstStat_ExitProg", LstStat_ExitProg)
        client.publish(f"{filename}/LstStat_LogicLoop", LstStat_LogicLoop)

        client.on_subscribe = on_subscribe

        # Subscribe to incoming Model_ID from Host
        client.subscribe(f"{line_id}/Model_ID")

        LstStat_MQTT = "Good"
        # Update const_data to store Model_ID
        update_ConstData()
        # Re-load or re-initialize models for new Model_ID
        reload_models_for_new_model_id()
    else:
        print(f"Failed to connect, return code {rc}")
        LstStat_MQTT = "Bad"
        update_ConstData()

def on_subscribe(client, userdata, mid, granted_qos):
            print("Subscribed successfully. mid =", mid, "granted_qos =", granted_qos)

def on_message(client, userdata, msg):
    global Model_ID , std_time
    topic = msg.topic
    payload = msg.payload.decode()
    
    # If this is the Model_ID topic
    if topic.endswith("Model_ID"):
        Model_ID = str(payload).split('_')[0]
        std_time = str(payload).split('_')[1]
        print(f"Received new Model_ID: {Model_ID}")
        
        # Update const_data to store Model_ID
        update_ConstData()
        # Re-load or re-initialize models for new Model_ID
        reload_models_for_new_model_id()


def reload_models_for_new_model_id():
    """
    Reload interpreters from dataset/models/{Model_ID}/model_S{S}.tflite
    Then update them in globals() so the main loop uses the new models.
    """
    global MainSub, filename

    # For each main box S, load the corresponding TFLite model
    for S in [sublist[0] for sublist in MainSub]:
        # e.g. dataset/models/{Model_ID}/model_S{S}.tflite
        new_model_path = f"dataset/models/{Model_ID}/model_{filename}_S{S}.tflite"
        
        if not os.path.exists(new_model_path):
            print(f"[WARNING] {new_model_path} does not exist. Using default model.")
            new_model_path = "dataset/model_default.tflite"
        
        try:
            new_interpreter = tflite.Interpreter(model_path=new_model_path)
            new_interpreter.allocate_tensors()
            globals()[f'interpreter{S}'] = new_interpreter
            globals()[f'model_file{S}'] = new_model_path  # for reference
            print(f"Loaded new model for S={S}: {new_model_path}")
        except Exception as ex:
            print(f"Failed to load new model for S={S}: {ex}")
            # fallback
            new_model_path = "dataset/model_default.tflite"
            new_interpreter = tflite.Interpreter(model_path=new_model_path)
            new_interpreter.allocate_tensors()
            globals()[f'interpreter{S}'] = new_interpreter
            globals()[f'model_file{S}'] = new_model_path

def on_publish(client, userdata, mid):
    print("Message published.")

client = mqtt.Client(protocol=mqtt.MQTTv311)
client.on_connect = on_connect
client.on_message = on_message
client.on_publish = on_publish

def on_log(client, userdata, level, buf):
    print("LOG:", level, buf)

client.on_log = on_log


# WiFi connection initiation and re-connection
# def connect_to_wifi(ssid, password):
#     global LstStat_WiFi
#     try:
#         # Check if the WiFi is already connected
#         check_connection = subprocess.run(['nmcli', '-t', '-f', 'ACTIVE,SSID', 'dev', 'wifi'], capture_output=True, text=True)
#         connected = any([line.split(':')[0] == 'yes' and line.split(':')[1] == ssid for line in check_connection.stdout.splitlines()])

#         if not connected:
#             print(f"Connecting to {ssid}...")
#             subprocess.run(['nmcli', 'dev', 'wifi', 'connect', ssid, 'password', password])
#             time.sleep(10)  # Wait for a few seconds to ensure the connection is established
#             LstStat_WiFi = "Good"
#         else:
#             print(f"Already connected to {ssid}.")
#             LstStat_WiFi = "Good"

#     except Exception as e:
#         print(f"Failed to connect to {ssid}. Error: {e}")
#         LstStat_WiFi = "Bad"

# def check_and_reconnect():
#     global LstStat_WiFi
#     while True:
#         # Check WiFi connection status
#         connection_status = subprocess.run(['nmcli', '-t', '-f', 'DEVICE,STATE', 'dev'], capture_output=True, text=True)
#         connected = any(['connected' in line for line in connection_status.stdout.splitlines()])

#         if not connected or LstStat_WiFi == "Bad":
#             LstStat_WiFi = "Bad"
#             connect_to_wifi(WiFi_SSID, WiFi_PWD)
#         else:
#             LstStat_WiFi = "Good"

def connect_to_wifi(ssid, username, password, iface):
    global LstStat_WiFi
    try:
        # Delete any existing connection with the same name
        subprocess.run(['sudo','nmcli', 'connection', 'delete', 'id', ssid], check=False)

        # Add a new connection with the specified parameters
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
        
        # Bring up the new connection
        subprocess.run(['sudo','nmcli', 'connection', 'up', ssid], check=True)
        
        LstStat_WiFi = "Good"
        print("Connected to WiFi.")
    except subprocess.CalledProcessError as e:
        #print(f"Failed to connect to WiFi. Error: {e}")
        LstStat_WiFi = "Bad"

def check_and_reconnect():
    global LstStat_WiFi
    while True:
        # Check WiFi connection status
        connection_status = subprocess.run(['sudo','nmcli', '-t', '-f', 'DEVICE,STATE', 'dev'], capture_output=True, text=True)
        connected = any(['connected' in line for line in connection_status.stdout.splitlines()])

        if not connected or LstStat_WiFi == "Bad":
            LstStat_WiFi = "Bad"
            connect_to_wifi(WiFi_SSID, WiFi_USERNAME, WiFi_PWD, interface_name)
        else:
            LstStat_WiFi = "Good"

def update_ConstData():
    global filename, File_Name, Device_Addr, Host_Addr, WiFi_SSID, WiFi_USERNAME, WiFi_PWD
    global Camera_Addr, LstStat_Cam, LstStat_PwdOff, LstStat_WiFi, LstStat_MQTT
    global LstStat_ExitProg, LstStat_LogicLoop, Model_ID, step

    ConstData_filename = f"dataset/const_data_{filename}.csv"
    Const_data = [[
        File_Name,
        Device_Addr,
        Host_Addr,
        WiFi_SSID,
        WiFi_USERNAME,
        WiFi_PWD,
        Camera_Addr,
        LstStat_Cam,
        LstStat_PwdOff,
        LstStat_WiFi,
        LstStat_MQTT,
        LstStat_ExitProg,
        LstStat_LogicLoop,
        Model_ID,
        step  # <--- Added here
    ]]

    with open(ConstData_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['File_Name','Device_Addr','Host_Addr','WiFi_SSID','WiFi_USERNAME','WiFi_PWD',
                         'Camera_Addr','LstStat_Cam','LstStat_PwdOff','LstStat_WiFi','LstStat_MQTT',
                         'LstStat_ExitProg','LstStat_LogicLoop','Model_ID','Step'])  # <--- New column
        writer.writerows(Const_data)


def get_camera_indices():
    # Execute the command and get the output
    result = subprocess.run(['v4l2-ctl', '--list-devices'], stdout=subprocess.PIPE)
    output = result.stdout.decode('utf-8')

    # Use regular expression to find all video addresses
    matches = re.findall(r'/dev/video(\d+)', output)
    
    # Convert matches to integers
    indices = [int(match) for match in matches]

    # Ensure 5 is first if present, then sort the remaining
    if 5 in indices:
        indices.remove(5)
        indices = [5] + sorted(indices)
    else:
        indices = sorted(indices)

    return indices

def get_ip_address(interfaces):
    for interface in interfaces:
        try:
            # Execute the ip command and get the output
            result = subprocess.run(['ip', 'addr', 'show', interface], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            # Check if the command was successful
            if result.returncode != 0:
                print(f"Error running ip addr show on {interface}: {result.stderr}")
                continue
            
            # Use regex to find the IP address
            match = re.search(r'inet (\d+\.\d+\.\d+\.\d+)', result.stdout)
            
            if match:
                return match.group(1)
        except Exception as e:
            print(f"An error occurred with interface {interface}: {e}")
            continue
    
    print("No IP address found for any interface")
    return None

# Function to get spots boxes from the mask
def get_spots_boxes(connected_components):
    (totalLabels, label_ids, values, centroid) = connected_components
    slots = []
    coef = 1
    for i in range(1, totalLabels):
        x1 = int(values[i, cv2.CC_STAT_LEFT] * coef)
        y1 = int(values[i, cv2.CC_STAT_TOP] * coef)
        w = int(values[i, cv2.CC_STAT_WIDTH] * coef)
        h = int(values[i, cv2.CC_STAT_HEIGHT] * coef)
        slots.append([x1, y1, w, h])
    return slots

def calc_diff(im1, im2):
    return np.abs(np.mean(im1) - np.mean(im2))

def empty_or_not(spot_bgr, interpreter, input_details, output_details):
    h_in, w_in = input_details[0]['shape'][1:3]
    img = cv2.cvtColor(spot_bgr, cv2.COLOR_BGR2RGB)           # BGR → RGB
    h0, w0 = img.shape[:2]
    s   = min(w_in / w0, h_in / h0)
    nw, nh = int(round(w0 * s)), int(round(h0 * s))
    img = cv2.resize(img, (nw, nh))
    canvas = np.full((h_in, w_in, 3), 114, dtype=np.uint8) # 114 = Ultralytics padding colour
    dw, dh = (w_in - nw) // 2, (h_in - nh) // 2
    canvas[dh:dh + nh, dw:dw + nw] = img

    if input_details[0]['dtype'] == np.uint8:
        tensor = canvas.astype(np.uint8)
    else:                           # handles FP16 / FP32 if you swap later
        tensor = canvas.astype(np.float32) / 255.0
    img_input =  np.expand_dims(tensor, 0)          # batch-dim + info

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], img_input)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index']) 

    output_dtype = output_details[0]['dtype']
    if output_dtype == np.uint8:
        output_scale, output_zero_point = output_details[0]['quantization']
        output_data = output_scale * (output_data - output_zero_point)
        pass

    predicted_class = np.argmax(output_data, axis=1)[0]
    return predicted_class

# Read csv file to get Main box, Sub box data structure
def GetStructureFromCSV(file_path):
    # Initialize a dictionary to store the count of sub boxes for each main box
    main_box_counts = {}
    # Open the CSV file
    with open(file_path, mode='r') as file:
        csv_reader = csv.reader(file)
        # Skip the title row
        next(csv_reader)
        # Read each row in the CSV file
        for row in csv_reader:
            main_box = int(row[0])
            if main_box in main_box_counts:
                main_box_counts[main_box] += 1
            else:
                main_box_counts[main_box] = 1

    result = [[main_box, count] for main_box, count in main_box_counts.items()]
    return result

def GetConstDataFromCSV(file_path):
    global File_Name, Device_Addr, Host_Addr, WiFi_SSID, WiFi_USERNAME, WiFi_PWD
    global Camera_Addr, LstStat_Cam, LstStat_PwdOff, LstStat_WiFi, LstStat_MQTT
    global LstStat_ExitProg, LstStat_LogicLoop, Model_ID, step

    with open(file_path, mode='r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # skip header
        for row in csv_reader:
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
            step = int(row[14]) if len(row) > 14 else 30  # fallback default

    return [[
        File_Name, Device_Addr, Host_Addr, WiFi_SSID, WiFi_USERNAME, WiFi_PWD,
        Camera_Addr, LstStat_Cam, LstStat_PwdOff, LstStat_WiFi, LstStat_MQTT,
        LstStat_ExitProg, LstStat_LogicLoop, Model_ID, step
    ]]


def create_config_data(spots, main_box, model_file):
    config_data = []
    for i, spot in enumerate(spots):
        x, y, w, h = spot
        sub_box = i + 1  # Sub box index starts from 1
        config_data.append([main_box, sub_box, model_file, x, y, w, h])
    return config_data

def create_box_mask(boxes, image_size, background_color, output_path):
    image = Image.new("RGB", image_size, background_color)
    draw = ImageDraw.Draw(image)

    # Draw each box on the image
    for box in boxes:
        x, y, w, h = box
        draw.rectangle([x, y, x + w, y + h], fill=(255, 255, 255))  # White filled rectangle

    # Save the image to the specified output path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image.save(output_path)
    #print(f"Image saved to {output_path}")

def boxes_overlap(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    if (x1 < x2 + w2 and x1 + w1 > x2 and
            y1 < y2 + h2  and y1 + h1 > y2):
        return True
    return False

def any_boxes_overlap(boxes):
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes_overlap(boxes[i], boxes[j]):
                return True
    return False

def resolve_other(prev_state: int, curr_state: int) -> int:
    """
    Map OTHER to the previous state so that EMPTY→OTHER behaves as EMPTY,
    and NOT_EMPTY→OTHER behaves as NOT_EMPTY.
    """
    if curr_state == OTHER:
        return prev_state if prev_state in (EMPTY, NOT_EMPTY) else EMPTY
    return curr_state


filename = os.path.basename(__file__)
parts = filename.split('_')
line_id = parts[1]
filename = filename.split('_')[1] + '_' + filename.split('_')[2]
File_Name = filename

# Connect MQTT
try:
    client.connect(Host_Addr, firewall_port, 60)
    client.loop_start()
except Exception as e:
    print("MQTT connection failed:", e)
    LstStat_MQTT = "Bad"
    update_ConstData()

MainSub = GetStructureFromCSV(f"dataset/config_data_{filename}.csv")

#Get constant data from CSV
Const_data = GetConstDataFromCSV(f"dataset/const_data_{filename}.csv")

#Update neccesary consant data from device

 # Get the latest camera address
Camera_Addr = get_camera_indices()[0]

#Camera_Addr ='http://127.0.0.1/video_feed'

#Camera_Addr ='rtsp://admin:admin1234@192.168.1.44:554/cam/realmonitor?channel=1&subtype=0'

print("Found camera address:", Camera_Addr)

 # Get the latest ip address
Device_Addr = get_ip_address(interfaces)

if Device_Addr:
    print(f"IP address found: {Device_Addr}")
else:
    print(f"Failed to get IP address for any interface")

update_ConstData()

# WiFi connection initiation
#connect_to_wifi(WiFi_SSID, WiFi_PWD)
connect_to_wifi(WiFi_SSID, WiFi_USERNAME, WiFi_PWD, interface_name)

# Start the WiFi check and reconnect thread
wifi_thread = threading.Thread(target=check_and_reconnect,daemon=True)
wifi_thread.start()

#Get the lastest config data from directory file
Config_data = []

for S in [sublist[0] for sublist in MainSub]:
    model_path = f"dataset/model_{filename}_S{S}.tflite"

    if os.path.exists(model_path):
        try:
            interpreter = tflite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
        except Exception as e:
            print(f"Failed to load model for S={S}: {e}")
            continue
    else:
        model_path = 'dataset/model_default.tflite'
        interpreter = tflite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()

    mask = f"dataset/mask_img_{filename}_S{S}.png"
    model_file = model_path.split('/', 1)[-1]
    mask_img = cv2.imread(mask, 0)
    spots = get_spots_boxes(cv2.connectedComponentsWithStats(mask_img, 4, cv2.CV_32S))

    spots_status = [None for _ in spots]
    #diffs = [None for _ in spots]
    spots_raw_status    = [None for _ in spots]      # NEW – pure model output
    previous_spots_status = [None for _ in spots]
    original_spots = spots.copy()

    # Create Config_data entries for spots1 and spots2
    Config_data.extend(create_config_data(spots, S, model_file))

    # Dynamically create variable names like model1, model2, etc.
    #globals()[f'model{S}'] = model
    globals()[f'interpreter{S}'] = interpreter

    globals()[f'mask{S}'] = mask
    globals()[f'model_file{S}'] = model_file
    globals()[f'mask{S}_img'] = mask_img
    globals()[f'spots{S}'] = spots

    globals()[f'spots_status{S}'] = spots_status
    #globals()[f'diffs{S}'] = diffs
    globals()[f'spots_raw_status{S}']  = spots_raw_status   # NEW
    globals()[f'previous_spots_status{S}'] = previous_spots_status
    globals()[f'original_spots{S}'] = original_spots

cap = cv2.VideoCapture(Camera_Addr)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

mask_height = frame_height
mask_width = frame_width
image_size = (frame_width, frame_height)

start_time = time.time()
t_E = t_S = t_prev_E = None
e_detected = s_detected = False

previous_frame = None
cycle_time = "N/A"
assembly_time = "N/A"

# Tkinter GUI setup
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
        self.is_image_capturing = False  # Add this line
        self.image_capture_folder = None  # Folder where images will be saved


    def create_widgets(self):
        self.master.geometry("320x280+1+1")

        # Create container frame+
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
            video_filename = f"dataset/cam_video_{filename}_{current_datetime}.mp4"
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
            messagebox.showinfo("Message", f"Sub Boxes are overlaping on Main Box No. {overlapping_sublist[0]}\nPlease check the overlaping again before saving.")

        else:
                Config_data.clear()  # Clear existing data
                for S in [sublist[0] for sublist in MainSub]:
                    Config_data.extend(create_config_data(globals()[f'spots{S}'], S, globals()[f'model_file{S}']))

                for S in [sublist[0] for sublist in MainSub]:
                    create_box_mask(globals()[f'spots{S}'], image_size, background_color, globals()[f'mask{S}'])

                for S in [sublist[0] for sublist in MainSub]:
                    globals()[f'original_spots{S}'] = globals()[f'spots{S}'].copy()

                csv_filename = f"dataset/config_data_{filename}.csv"
                # Check if the file already exists and delete it if it does
                if os.path.exists(csv_filename):
                    try:
                        os.remove(csv_filename)
                        #print(f"{csv_filename} has been removed successfully.")
                    except PermissionError as e:
                        print(f"Error: {e}")

                # Write Config_data to CSV
                with open(csv_filename, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(['Main Box', 'Sub Box', 'Model File', 'X', 'Y', 'W', 'H'])  # Write header
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
            # Handle creation of new main box
            new_main_index = max([i[0] for i in MainSub]) + 1
            MainSub.append([new_main_index, 0])

            # Initialize necessary globals for the new main box
            globals()[f'model{new_main_index}'] = None  # Example: Initialize model as None
            globals()[f'spots{new_main_index}'] = []
            globals()[f'spots_status{new_main_index}'] = []
            globals()[f'original_spots{new_main_index}'] = []
            globals()[f'diffs{new_main_index}'] = []
            globals()[f'model_file{new_main_index}'] = 'dataset/model_default'
            globals()[f'mask{new_main_index}'] = f"dataset/mask_img_{filename}_S{new_main_index}.png"

            # Create an example image for the mask
            image = Image.new("RGB", image_size, background_color)

            # Save the image to the specified mask path
            os.makedirs(os.path.dirname(globals()[f'mask{new_main_index}']), exist_ok=True)
            image.save(globals()[f'mask{new_main_index}'])

            # Update choices list with new main box index
            choices = [str(sublist[0]) for sublist in MainSub] + ["New"]
            adjust_page.selected_main.set(choices[0])  # Reset dropdown to first option

            # Update OptionMenu with new choices
            menu = adjust_page.dropdown["menu"]
            menu.delete(0, "end")  # Clear existing options
            for choice in choices:
                menu.add_command(label=choice, command=tk._setit(adjust_page.selected_main, choice))

            spots_list = globals()[f'spots{new_main_index}']
            selected_main = new_main_index
        else:
            selected_main = int(selected_main)
            spots_list = globals().get(f'spots{selected_main}', [])

        # Add a new spot to spots_list
        if spots_list:
            first_spot = spots_list[0]  # Assuming spots_list is not empty
            new_spot = [0, 0, first_spot[2], first_spot[3]]  # Example new spot
        else:
            new_spot = default_spot  # Default new spot dimensions

        spots_list.append(new_spot)

        # Update the global lists
        globals()[f'spots{selected_main}'] = spots_list
        globals()[f'spots_status{selected_main}'].append(new_spot)
        globals()[f'diffs{selected_main}'].append(new_spot)

        self.save_adjustments()  # Save adjustments or changes

    def delete_box(self):
        global MainSub, filename
        adjust_page = self.frames["AdjustPage"]
        selected_main = adjust_page.selected_main.get()
        selected_main_int = int(selected_main)

        selected_index = globals()[f'selected_object_index{selected_main_int}']
        if selected_index is not None:
            spots_list = globals()[f'spots{selected_main_int}']
            spots_list.pop(selected_index)
            globals()[f'spots_status{selected_main_int}'].pop(selected_index)
            globals()[f'diffs{selected_main_int}'].pop(selected_index)
            globals()[f'previous_spots_status{selected_main_int}'].pop(selected_index)
            globals()[f'original_spots{selected_main_int}'].pop(selected_index)
            globals()[f'selected_object_index{selected_main_int}'] = None

            self.save_adjustments()
            print(f"Deleted spot at index {selected_index} from Main box {selected_main_int}")

            MainSub = GetStructureFromCSV(f"dataset/config_data_{filename}.csv")
            print(MainSub)
            # Update OptionMenu with new choices
            menu = adjust_page.dropdown["menu"]
            menu.delete(0, "end")  # Clear existing options
            choices = [str(sublist[0]) for sublist in MainSub] + ["New"]
            for choice in choices:
                menu.add_command(label=choice, command=tk._setit(adjust_page.selected_main, choice))
        else:
            print("No box selected to delete.")

    def delete_main_box(self):
        global MainSub
        adjust_page = self.frames["AdjustPage"]
        selected_main = adjust_page.selected_main.get()
        selected_main_int = int(selected_main) - 1

        if 0 <= selected_main_int < len(MainSub):
            # Pop-up message to confirm deletion
            confirm = messagebox.askyesno("Confirm Deletion", f"The deletion can't be undo.\nAre you sure to delete the Main Box No. {selected_main} ?")
            if confirm:
                # Delete the item at selected_main_int index from MainSub
                del MainSub[selected_main_int]
                self.save_adjustments()
                print(f"Deleted from Main box {selected_main_int}")
                choices = [str(sublist[0]) for sublist in MainSub] + ["New"]
                adjust_page.selected_main.set(choices[0])  # Reset dropdown to first option

                # Update OptionMenu with new choices
                menu = adjust_page.dropdown["menu"]
                menu.delete(0, "end")  # Clear existing options
                for choice in choices:
                    menu.add_command(label=choice, command=tk._setit(adjust_page.selected_main, choice))
        else:
            print(f"Index {selected_main_int} is out of range.")

    def update_treeview(self):
        if hasattr(self, 'my_table') and self.my_table:
            # Clear existing items in the Treeview
            for item in self.my_table.get_children():
                self.my_table.delete(item)

            # Insert new data from Config_data
            for i, entry in enumerate(Config_data):
                main_box, sub_box, model_file, x, y, w, h = entry
                self.my_table.insert(parent='', index='end', iid=f'entry_{i}', text='',
                                     values=(main_box, sub_box, model_file, x, y, w, h))

    def configuration(self):
        def create_table_columns(table, columns):
            table['columns'] = columns
            for col in columns:
                table.column(col, anchor=tk.CENTER,
                             width=50 if col not in ['main_box', 'sub_box', 'model_file'] else 70 if col in ['main_box',
                                                                                                             'sub_box'] else 130)
                table.heading(col, text=col.replace('_', ' ').title(), anchor=tk.CENTER)
            table.column("#0", width=0, stretch=tk.NO)
            table.heading("#0", text="", anchor=tk.CENTER)

        self.config_window = tk.Toplevel(self.master)
        self.config_window.title("Configuration Data")
        self.config_window.geometry('490x200+1+220') ###
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
        # Show Index Button
        self.show_index_button = tk.Button(self, text="Display Index", command=self.controller.show_index, padx=internal_padx, pady=internal_pady)
        self.show_index_button.grid(row=0, column=0, padx=external_padx, pady=external_pady)

        # Record Video Button
        self.record_button = tk.Button(self, text="Video Recorder",
                                       command=lambda: self.controller.show_frame("RecordPage"), padx=internal_padx, pady=internal_pady)
        self.record_button.grid(row=1, column=0, padx=external_padx, pady=external_pady)

        # Adjust Position Button
        self.adjust_button = tk.Button(self, text="Box Editer Mode",
                                       command=lambda: self.controller.show_frame("AdjustPage"), padx=internal_padx, pady=internal_pady)
        self.adjust_button.grid(row=0, column=1, padx=external_padx, pady=external_pady)

        # Configuration Button
        self.config_button = tk.Button(self, text="Configuration Data", command=self.controller.configuration, padx=internal_padx, pady=internal_pady)
        self.config_button.grid(row=1, column=1, padx=external_padx, pady=external_pady)

class AdjustPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.selected_main = tk.StringVar(self)
        self.create_widgets()

    def create_widgets(self):
        # Extract the first numbers from MainSub
        choices = [str(sublist[0]) for sublist in MainSub]+ ["New"]
        self.selected_main.set(choices[0])  # Default selection
        self.dropdown = tk.OptionMenu(self, self.selected_main, *choices)

        self.dropdown.grid(row=1, column=0, padx=external_padx, pady=external_pady)

        text_main_selection = tk.Label(self, text="[Select Main Box No.]\n  V\n  V",font=("Helvetica", 9, "bold"))
        text_main_selection.grid(row=0, column=0)

        add_button = tk.Button(self, text="(+)Add Main/Sub Box", command=self.controller.add_box, padx=internal_padx, pady=internal_pady)
        add_button.grid(row=0, column=1, padx=external_padx, pady=external_pady)

        delete_button = tk.Button(self, text="(-)Delete Sub Box", command=self.controller.delete_box, padx=internal_padx, pady=internal_pady)
        delete_button.grid(row=2, column=1, padx=external_padx, pady=external_pady)

        delete_mainbutton = tk.Button(self, text="(-)Delete Main Box", command=self.controller.delete_main_box, padx=internal_padx, pady=internal_pady)
        delete_mainbutton.grid(row=1, column=1, padx=external_padx, pady=external_pady)

        # Back Button to return to the StartPage
        back_button = tk.Button(self, text="<<Back", command=lambda: self.controller.show_frame("StartPage"), padx=internal_padx, pady=internal_pady)
        back_button.grid(row=2, column=0, padx=external_padx, pady=external_pady)

        # Save Button
        save_button = tk.Button(self, text="Save", command=self.controller.save_adjustments, padx=internal_padx, pady=internal_pady)
        save_button.grid(row=0, column=2, padx=external_padx, pady=external_pady)

        # Reset Button
        reset_button = tk.Button(self, text="Reset", command=self.controller.reset_adjustments, padx=internal_padx, pady=internal_pady)
        reset_button.grid(row=1, column=2, padx=external_padx, pady=external_pady)

class RecordPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.create_widgets()

    def create_widgets(self):
        # Start Video Recording Button
        start_video_button = tk.Button(self, text="Start Video Recording", command=self.controller.start_recording,
                                 padx=internal_padx, pady=internal_pady)
        start_video_button.pack(padx=external_padx, pady=external_pady)

        # Stop Video Recording Button
        stop_video_button = tk.Button(self, text="Stop Video Recording", command=self.controller.stop_recording, padx=internal_padx,
                                pady=internal_pady)
        stop_video_button.pack(padx=external_padx, pady=external_pady)

        # Start Image Capture Button
        start_image_button = tk.Button(self, text="Start Image Capture", command=self.controller.start_image_capture,
                                       padx=internal_padx, pady=internal_pady)
        start_image_button.pack(padx=external_padx, pady=external_pady)

        # Stop Image Capture Button
        stop_image_button = tk.Button(self, text="Stop Image Capture", command=self.controller.stop_image_capture,
                                      padx=internal_padx, pady=internal_pady)
        stop_image_button.pack(padx=external_padx, pady=external_pady)

        # Back Button to return to the StartPage
        back_button = tk.Button(self, text="<<Back", command=lambda: self.controller.show_frame("StartPage"),
                                padx=internal_padx, pady=internal_pady)
        back_button.pack(padx=external_padx, pady=external_pady)


def update_gui():
    global app
    root = tk.Tk()
    root.title("Control Panel")
    app = Application(master=root)
    app.mainloop()

gui_thread = threading.Thread(target=update_gui)
gui_thread.start()

frame_nmr = 0
ret = True
cycle_counter = 1
cycle_time_start = time.time()

for S in [sublist[0] for sublist in MainSub]:
    globals()[f'selected_object_index{S}'] = None
    dragging = False
    resizing = False
    resizing_edge = None

def mouse_events(event, x, y, flags, param):
    global dragging, resizing, resizing_edge

    # Get the actual frame size
    actual_width = frame.shape[1]
    actual_height = frame.shape[0]

    # Get the resized window size
    window_name = f'Real-Time Monitor ({filename})'
    window_x, window_y, window_w, window_h = cv2.getWindowImageRect(window_name)
    
    # Calculate scaling factors
    scale_x = actual_width / window_w
    scale_y = actual_height / window_h
    
    # Adjust mouse coordinates to match actual frame size
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

                    # Check if click is on the edges for resizing
                    if (left_edge - resize_margin <= x <= left_edge + resize_margin and
                            top_edge <= y <= bottom_edge):
                        globals()[f'selected_object_index{S}'] = i
                        resizing = True
                        resizing_edge = 'left'
                        break
                    elif (right_edge - resize_margin <= x <= right_edge + resize_margin and
                          top_edge <= y <= bottom_edge):
                        globals()[f'selected_object_index{S}'] = i
                        resizing = True
                        resizing_edge = 'right'
                        break
                    elif (top_edge - resize_margin <= y <= top_edge + resize_margin and
                          left_edge <= x <= right_edge):
                        globals()[f'selected_object_index{S}'] = i
                        resizing = True
                        resizing_edge = 'top'
                        break
                    elif (bottom_edge - resize_margin <= y <= bottom_edge + resize_margin and
                          left_edge <= x <= right_edge):
                        globals()[f'selected_object_index{S}'] = i
                        resizing = True
                        resizing_edge = 'bottom'
                        break

                # If not resizing, check for dragging
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
    """
    Put the current Model_ID on the top-left corner or wherever appropriate.
    """
    global Model_ID
    text_pos = (10, 30)  # X=10, Y=30
    cv2.putText(frame, f"Model: {Model_ID}", text_pos,
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)


while ret:
    ret, frame = cap.read()
    LstStat_Cam = "Good"

    if not ret:
        LstStat_Cam = "Bad"
        update_ConstData()
        break

    copy_frame = frame.copy()
    ## Text position
    adjust_text_position = (frame.shape[1] - 170, 65)  # Position for adjust text
    time_text_position = (frame.shape[1] - 360, 32)  # Position for time text
    '''
    if frame_nmr % step == 0 and previous_frame is not None:
        for S in [sublist[0] for sublist in MainSub]:
            for spot_indx, spot in enumerate(globals()[f'spots{S}']):
                x1, y1, w, h = spot
                spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
                globals()[f'diffs{S}'][spot_indx] = calc_diff(spot_crop, previous_frame[y1:y1 + h, x1:x1 + w, :])
    '''
    if frame_nmr % step == 0:
        '''
        if previous_frame is None:
            LstStat_LogicLoop = "Bad"
            for S in [sublist[0] for sublist in MainSub]:
                arr_ = range(len(globals()[f'spots{S}']))
                globals()[f'arr_{S}'] = arr_

        else:
            for S in [sublist[0] for sublist in MainSub]:
                diffs = globals()[f'diffs{S}']
                arr_ = [j for j in np.argsort(diffs) if diffs[j] / np.amax(diffs) > diff_threshold]

                globals()[f'arr_{S}'] = arr_
        '''
        for S in [sublist[0] for sublist in MainSub]:
            model_path = f"dataset/model_{filename}_S{S}.tflite"

            if os.path.exists(model_path):
                try:
                    interpreter = tflite.Interpreter(model_path=model_path)
                    interpreter.allocate_tensors()
                except Exception as e:
                    print(f"Failed to load model for S={S}: {e}")
                    continue
            else:
                model_path = 'dataset/model_default.tflite'
                interpreter = tflite.Interpreter(model_path=model_path)
                interpreter.allocate_tensors()

            # Store the interpreter in globals for access in other parts of the code
            globals()[f'interpreter{S}'] = interpreter

        # Before the main loop, cache the details
        for S in [sublist[0] for sublist in MainSub]:
            interpreter = globals().get(f'interpreter{S}', None)
            if interpreter is None:
                continue
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            globals()[f'input_details{S}'] = input_details
            globals()[f'output_details{S}'] = output_details

            spots = globals()[f'spots{S}']
            # now classify every spot each cycle
            for spot_indx, spot in enumerate(spots):
                x1, y1, w, h = spot
                spot_crop = frame[y1:y1 + h, x1:x1 + w, :]

                #spot_status = empty_or_not(
                raw_state = empty_or_not(

                    spot_crop,
                    globals()[f'interpreter{S}'],
                    globals()[f'input_details{S}'],
                    globals()[f'output_details{S}']
                )
                
                # NEW ────────────────────────────────────────────────
                globals()[f'spots_raw_status{S}'][spot_indx] = raw_state   # keep for colour
                prev_state = globals()[f'previous_spots_status{S}'][spot_indx]
                resolved_state = resolve_other(prev_state, raw_state)
                # ────────────────────────────────────────────────────

                #globals()[f'spots_status{S}'][spot_indx] = spot_status
                globals()[f'spots_status{S}'][spot_indx] = resolved_state


        #Below is logic for a specific application

        ##Logic section
        # global spots_status1 
        global spots_status1
        # global previous_spots_status1
        global previous_spots_status1

        not_empty_count1 = sum(1 for s in spots_status1 if s == NOT_EMPTY)
        #other_detected = any(s == OTHER for s in spots_status1)
        previous_not_empty_count1 = sum(1 for s in previous_spots_status1 if s == NOT_EMPTY)
        #previous_other_detected = any(s == OTHER for s in previous_spots_status1)

        #if not other_detected:
        if not e_detected and not_empty_count1 > previous_not_empty_count1:
            #if not e_detected and not_empty_count1 > previous_not_empty_count1 and not previous_other_detected:
            t_E = time.time() - start_time
            e_detected = True
            LstStat_LogicLoop = "Good"
            print("E stage detected at: {:.1f} seconds".format(t_E))

            if t_prev_E is not None:
                cycle_time = (t_E - t_prev_E)
                e_detected = False
                print("Cycle time: {}".format(cycle_time))
                if cycle_time > float(std_time)*0.4:
                    client.publish(f"{filename}/cycle_time", "{:.1f}".format(cycle_time))
                    t_prev_E = t_E
            else:
                e_detected = False
                t_prev_E = t_E

        #Above is logic for a specific application

        previous_frame = frame.copy()

        for S in [sublist[0] for sublist in MainSub]:
            spots_status = globals()[f'spots_status{S}']
            globals()[f'previous_spots_status{S}'] = spots_status.copy()

    display_frame = frame.copy()

    # Show model ID at top-left
    show_model_id_on_frame(display_frame)

    for S in [sublist[0] for sublist in MainSub]:
        spots = globals()[f'spots{S}']
        #spots_status = globals()[f'spots_status{S}']
        spots_status = globals()[f'spots_raw_status{S}']    # new – raw, may be OTHER

        for spot_indx, spot in enumerate(spots):
            spot_status = spots_status[spot_indx]
            x1, y1, w, h = spots[spot_indx]
            if spot_status == EMPTY:
                color = empty_color
            elif spot_status == NOT_EMPTY:
                color = not_empty_color
            else:
                color = other_color
            display_frame = cv2.rectangle(display_frame, (x1, y1), (x1 + w, y1 + h), color, 2)
            if app.show_idx:
                texts = str(S) + "/" + str(spot_indx + 1)
                textDim = cv2.getTextSize(texts, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.putText(display_frame, texts, (int(x1 + w // 2 - (textDim[0][0])/2), int(y1 + h // 2 + (textDim[0][1])/2)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
            if globals()[f'selected_object_index{S}'] == spot_indx:
                cv2.rectangle(display_frame, (x1 - 2, y1 - 2), (x1 + w + 2, y1 + h + 2), selected_color, 3)

    current_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")


    if app.is_adjust:
        cv2.putText(display_frame, "Box Editor Mode", adjust_text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    adjust_text_color, 2, cv2.LINE_AA)

    cv2.putText(display_frame, current_time, time_text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2, cv2.LINE_AA)

    cv2.namedWindow(f'Real-Time Monitor ({filename})', cv2.WINDOW_NORMAL)
    
    cv2.setWindowProperty(f'Real-Time Monitor ({filename})', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    display_frame = cv2.resize(display_frame,(960,720))

    cv2.imshow(f'Real-Time Monitor ({filename})', display_frame)

    if app is not None and app.is_recording:
        if app.video_writer is not None:
            app.video_writer.write(copy_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        LstStat_ExitProg = "Good"
        update_ConstData()
        break

    frame_nmr += 1
    LstStat_PwdOff = "Bad"

    # Image capture logic
    if app is not None and app.is_image_capturing:
        N = 150 # Set your desired frame interval here
        if frame_nmr % N == 0:
            # For each spot, save the image
            for S in [sublist[0] for sublist in MainSub]:
                spots = globals()[f'spots{S}']
                for spot_indx, spot in enumerate(spots):
                    x1, y1, w, h = spot
                    spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
                    # Create subfolder for each spot
                    spot_folder = os.path.join(app.image_capture_folder, f"Spot_{S}_{spot_indx+1}")
                    os.makedirs(spot_folder, exist_ok=True)
                    # Save image
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    image_filename = os.path.join(spot_folder, f"frame_{frame_nmr}_{timestamp}.jpg")
                    cv2.imwrite(image_filename, spot_crop)
                    print(f"Saved image: {image_filename}")

LstStat_PwdOff = "Good"
update_ConstData()
# Once finished, stop loop gracefully
client.loop_stop()
client.disconnect()

cap.release()
cv2.destroyAllWindows()
app.master.after(0,app.master.quit)  # This will close the Tkinter window