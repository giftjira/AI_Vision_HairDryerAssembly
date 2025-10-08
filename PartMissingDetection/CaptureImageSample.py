import cv2
import os
from datetime import datetime
import tkinter as tk
from tkinter import ttk
from threading import Thread

class CameraCaptureApp:
    def __init__(self, root, camera_index=5):
        self.root = root
        self.camera_index = camera_index
        self.frame_interval = 60
        self.recording = False
        self.frame_count = 0
        self.cap = None
        self.thread = None
        self.running = False
        self.output_folder = None

        # Create GUI
        self.root.title("Camera Capture")
        self.root.geometry("300x200")

        self.start_button = ttk.Button(root, text="Start Recording", command=self.start_recording)
        self.start_button.pack(pady=10)

        self.stop_button = ttk.Button(root, text="Stop Recording", command=self.stop_recording, state=tk.DISABLED)
        self.stop_button.pack(pady=10)

        self.interval_label = ttk.Label(root, text="Capture Interval (frames):")
        self.interval_label.pack(pady=5)

        self.interval_entry = ttk.Entry(root)
        self.interval_entry.insert(0, "60")
        self.interval_entry.pack(pady=5)

        self.quit_button = ttk.Button(root, text="Quit", command=self.quit_app)
        self.quit_button.pack(pady=10)

    def start_recording(self):
        try:
            self.frame_interval = int(self.interval_entry.get())
        except ValueError:
            self.frame_interval = 60

        self.recording = True
        self.running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)

        # Set up the output folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_folder = f"record_{timestamp}"
        os.makedirs(self.output_folder, exist_ok=True)

        # Start camera
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print(f"Error: Unable to access camera at index {self.camera_index}")
            self.stop_recording()
            return

        self.thread = Thread(target=self.capture_frames, daemon=True)
        self.thread.start()

    def stop_recording(self):
        self.recording = False
        self.running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Recording stopped.")

    def capture_frames(self):
        print("Recording started.")
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Unable to read frame.")
                self.stop_recording()
                break

            # Show the real-time video feed
            cv2.imshow("Live Video Feed", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop_recording()
                break

            if self.recording:
                self.frame_count += 1
                if self.frame_count % self.frame_interval == 0:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = os.path.join(self.output_folder, f"capture_{timestamp}.jpg")
                    cv2.imwrite(filename, frame)
                    print(f"Saved: {filename}")

    def quit_app(self):
        self.stop_recording()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = CameraCaptureApp(root)
    root.mainloop()
