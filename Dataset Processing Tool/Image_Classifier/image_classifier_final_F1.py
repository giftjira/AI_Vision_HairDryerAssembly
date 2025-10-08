import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import shutil
from PIL import Image, ImageTk
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import cv2
from pathlib import Path
import threading
from datetime import datetime

class ImageSimilarityClassifier:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Similarity Classifier & Auto-Sorter")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Variables
        self.num_classes = tk.IntVar(value=2)
        self.prototype_images = {}
        self.source_folder = tk.StringVar()
        self.destination_folders = {}
        self.unclassified_folder = tk.StringVar()
        self.progress_var = tk.DoubleVar()
        self.status_var = tk.StringVar(value="พร้อมใช้งาน")
        self.current_operation = tk.StringVar()
        self.threshold = tk.DoubleVar(value=0.3)  # Similarity threshold
        
        # Threading control
        self.processing = False
        self.stop_processing = False
        
        # Style configuration
        self.setup_styles()
        self.setup_ui()

    def setup_styles(self):
        """Configure custom styles for better UX"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure custom styles
        style.configure('Title.TLabel', font=('Arial', 18, 'bold'), foreground='#2c3e50')
        style.configure('Subtitle.TLabel', font=('Arial', 12, 'bold'), foreground='#34495e')
        style.configure('Info.TLabel', font=('Arial', 10), foreground='#7f8c8d')
        style.configure('Success.TLabel', font=('Arial', 10), foreground='#27ae60')
        style.configure('Error.TLabel', font=('Arial', 10), foreground='#e74c3c')
        style.configure('Custom.TButton', font=('Arial', 10, 'bold'))
        style.configure('Action.TButton', font=('Arial', 12, 'bold'))
        
        # Configure frame styles
        style.configure('Card.TLabelframe', relief='raised', borderwidth=2)
        style.configure('Card.TLabelframe.Label', font=('Arial', 11, 'bold'), foreground='#2c3e50')

    def setup_ui(self):
        """Setup the main user interface"""
        # Main container with scrollbar
        main_canvas = tk.Canvas(self.root, bg='#f0f0f0')
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=main_canvas.yview)
        scrollable_frame = ttk.Frame(main_canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all"))
        )
        
        main_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        main_canvas.configure(yscrollcommand=scrollbar.set)
        
        main_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Main content frame
        main_frame = ttk.Frame(scrollable_frame, padding="20")
        main_frame.pack(fill="both", expand=True)
        
        # Title section
        title_frame = ttk.Frame(main_frame)
        title_frame.pack(fill="x", pady=(0, 30))
        
        ttk.Label(title_frame, text="🖼️ Image Similarity Classifier", 
                 style='Title.TLabel').pack()
        ttk.Label(title_frame, text="ระบบจัดแยกรูปภาพอัตโนมัติด้วย AI", 
                 style='Info.TLabel').pack(pady=(5, 0))
        
        # Step 1: Number of classes
        self.create_step1(main_frame)
        
        # Step 1.5: Similarity threshold
        self.create_threshold_section(main_frame)
        
        # Step 2: Prototype images
        self.create_step2(main_frame)
        
        # Step 3: Source folder
        self.create_step3(main_frame)
        
        # Step 4: Destination folders
        self.create_step4(main_frame)
        
        # Control buttons
        self.create_controls(main_frame)
        
        # Progress and status
        self.create_status_section(main_frame)
        
        # Initialize UI
        self.update_class_inputs()

    def create_step1(self, parent):
        """Create step 1 UI - Number of classes"""
        step1_frame = ttk.LabelFrame(parent, text="📊 ขั้นตอนที่ 1: กำหนดจำนวนประเภทรูปภาพ", 
                                   style='Card.TLabelframe', padding="15")
        step1_frame.pack(fill="x", pady=(0, 15))
        
        info_frame = ttk.Frame(step1_frame)
        info_frame.pack(fill="x", pady=(0, 10))
        ttk.Label(info_frame, text="เลือกจำนวนประเภทของรูปภาพที่ต้องการแยก (2-10 ประเภท)", 
                 style='Info.TLabel').pack(anchor="w")
        
        input_frame = ttk.Frame(step1_frame)
        input_frame.pack(fill="x")
        ttk.Label(input_frame, text="จำนวนประเภท:", font=('Arial', 10, 'bold')).pack(side="left", padx=(0, 10))
        
        spinbox = ttk.Spinbox(input_frame, from_=2, to=10, width=15, textvariable=self.num_classes, 
                            command=self.update_class_inputs, font=('Arial', 10))
        spinbox.pack(side="left")
        
        # Bind spinbox events
        spinbox.bind('<KeyRelease>', lambda e: self.root.after(100, self.update_class_inputs))

    def create_threshold_section(self, parent):
        """Create threshold adjustment section"""
        threshold_frame = ttk.LabelFrame(parent, text="⚙️ ปรับแต่งความไวในการจำแนก", 
                                       style='Card.TLabelframe', padding="15")
        threshold_frame.pack(fill="x", pady=(0, 15))
        
        info_frame = ttk.Frame(threshold_frame)
        info_frame.pack(fill="x", pady=(0, 10))
        
        info_text = ("ค่าความไว: 0.1-0.9 (ค่าต่ำ = เข้มงวดมาก, ค่าสูง = ผ่อนปรนมาก)\n"
                    "แนะนำ: 0.2-0.4 สำหรับรูปที่แตกต่างชัด, 0.4-0.6 สำหรับรูปที่คล้ายกัน")
        ttk.Label(info_frame, text=info_text, style='Info.TLabel').pack(anchor="w")
        
        control_frame = ttk.Frame(threshold_frame)
        control_frame.pack(fill="x")
        
        ttk.Label(control_frame, text="ค่าความไว:", font=('Arial', 10, 'bold')).pack(side="left", padx=(0, 10))
        
        threshold_scale = ttk.Scale(control_frame, from_=0.1, to=0.9, 
                                  variable=self.threshold, orient="horizontal", length=200)
        threshold_scale.pack(side="left", padx=(0, 10))
        
        self.threshold_label = ttk.Label(control_frame, text=f"{self.threshold.get():.2f}", 
                                       font=('Arial', 10, 'bold'), foreground='#e67e22')
        self.threshold_label.pack(side="left")
        
        # Update label when scale changes
        threshold_scale.configure(command=self.update_threshold_label)

    def create_step2(self, parent):
        """Create step 2 UI - Prototype images"""
        self.step2_frame = ttk.LabelFrame(parent, text="🎯 ขั้นตอนที่ 2: เลือกรูปตัวอย่างแต่ละประเภท", 
                                        style='Card.TLabelframe', padding="15")
        self.step2_frame.pack(fill="x", pady=(0, 15))
        
        info_label = ttk.Label(self.step2_frame, 
                              text="เลือกรูปภาพตัวอย่างที่เป็นตัวแทนของแต่ละประเภท (ควรเลือกรูปที่ชัดเจนและเป็นตัวแทนที่ดี)", 
                              style='Info.TLabel')
        info_label.pack(anchor="w", pady=(0, 10))

    def create_step3(self, parent):
        """Create step 3 UI - Source folder"""
        step3_frame = ttk.LabelFrame(parent, text="📁 ขั้นตอนที่ 3: เลือกโฟลเดอร์ต้นทาง", 
                                   style='Card.TLabelframe', padding="15")
        step3_frame.pack(fill="x", pady=(0, 15))
        
        ttk.Label(step3_frame, text="เลือกโฟลเดอร์ที่มีรูปภาพที่ต้องการจัดแยก", 
                 style='Info.TLabel').pack(anchor="w", pady=(0, 10))
        
        input_frame = ttk.Frame(step3_frame)
        input_frame.pack(fill="x")
        
        entry = ttk.Entry(input_frame, textvariable=self.source_folder, width=60, font=('Arial', 10))
        entry.pack(side="left", fill="x", expand=True, padx=(0, 10))
        
        ttk.Button(input_frame, text="📂 เลือกโฟลเดอร์", command=self.select_source_folder,
                  style='Custom.TButton').pack(side="right")

    def create_step4(self, parent):
        """Create step 4 UI - Destination folders"""
        self.step4_frame = ttk.LabelFrame(parent, text="🎯 ขั้นตอนที่ 4: เลือกโฟลเดอร์ปลายทางแต่ละประเภท", 
                                        style='Card.TLabelframe', padding="15")
        self.step4_frame.pack(fill="x", pady=(0, 15))
        
        ttk.Label(self.step4_frame, 
                 text="เลือกโฟลเดอร์ที่จะเก็บรูปภาพแต่ละประเภท และโฟลเดอร์สำหรับรูปที่ระบบไม่สามารถจัดประเภทได้", 
                 style='Info.TLabel').pack(anchor="w", pady=(0, 10))

    def create_controls(self, parent):
        """Create control buttons"""
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill="x", pady=20)
        
        button_frame = ttk.Frame(control_frame)
        button_frame.pack()
        
        self.start_btn = ttk.Button(button_frame, text="🚀 เริ่มจัดแยกรูปภาพ", 
                                   command=self.start_classification, style='Action.TButton')
        self.start_btn.pack(side="left", padx=(0, 15))
        
        self.stop_btn = ttk.Button(button_frame, text="⏹️ หยุดการทำงาน", 
                                  command=self.stop_classification, style='Custom.TButton', state='disabled')
        self.stop_btn.pack(side="left", padx=(0, 15))
        
        ttk.Button(button_frame, text="🔄 รีเซ็ต", command=self.reset_all,
                  style='Custom.TButton').pack(side="left")

    def create_status_section(self, parent):
        """Create status and progress section"""
        status_frame = ttk.LabelFrame(parent, text="📊 สถานะการทำงาน", 
                                    style='Card.TLabelframe', padding="15")
        status_frame.pack(fill="x", pady=(0, 20))
        
        # Current operation
        self.operation_label = ttk.Label(status_frame, textvariable=self.current_operation, 
                                       style='Info.TLabel')
        self.operation_label.pack(anchor="w", pady=(0, 5))
        
        # Progress bar
        self.progress_bar = ttk.Progressbar(status_frame, variable=self.progress_var, 
                                          maximum=100, length=400, mode='determinate')
        self.progress_bar.pack(fill="x", pady=(0, 5))
        
        # Status label
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var, 
                                    style='Success.TLabel')
        self.status_label.pack(anchor="w")

    def update_class_inputs(self):
        """Update UI based on number of classes"""
        # Clear existing widgets
        for widget in self.step2_frame.winfo_children()[1:]:  # Keep info label
            widget.destroy()
        for widget in self.step4_frame.winfo_children()[1:]:  # Keep info label
            widget.destroy()
        
        self.prototype_images.clear()
        self.destination_folders.clear()
        
        # Create prototype image inputs
        for i in range(self.num_classes.get()):
            # Create frame for each class
            class_frame = ttk.Frame(self.step2_frame)
            class_frame.pack(fill="x", pady=5)
            
            ttk.Label(class_frame, text=f"ประเภทที่ {i+1}:", 
                     font=('Arial', 10, 'bold'), width=12).pack(side="left", padx=(0, 10))
            
            var = tk.StringVar()
            self.prototype_images[i] = var
            
            entry = ttk.Entry(class_frame, textvariable=var, width=50, font=('Arial', 9))
            entry.pack(side="left", fill="x", expand=True, padx=(0, 10))
            
            ttk.Button(class_frame, text="🖼️ เลือกรูป", 
                      command=lambda idx=i: self.select_prototype(idx),
                      style='Custom.TButton').pack(side="right")
            
            # Preview label
            preview_label = ttk.Label(class_frame, text="", style='Info.TLabel')
            preview_label.pack(side="right", padx=(10, 0))
            
        # Create destination folder inputs
        for i in range(self.num_classes.get()):
            dest_frame = ttk.Frame(self.step4_frame)
            dest_frame.pack(fill="x", pady=5)
            
            ttk.Label(dest_frame, text=f"ประเภทที่ {i+1}:", 
                     font=('Arial', 10, 'bold'), width=12).pack(side="left", padx=(0, 10))
            
            dvar = tk.StringVar()
            self.destination_folders[i] = dvar
            
            entry = ttk.Entry(dest_frame, textvariable=dvar, width=50, font=('Arial', 9))
            entry.pack(side="left", fill="x", expand=True, padx=(0, 10))
            
            ttk.Button(dest_frame, text="📁 เลือกโฟลเดอร์", 
                      command=lambda idx=i: self.select_destination(idx),
                      style='Custom.TButton').pack(side="right")
        
        # Unclassified folder
        unclass_frame = ttk.Frame(self.step4_frame)
        unclass_frame.pack(fill="x", pady=(15, 5))
        
        ttk.Label(unclass_frame, text="ไม่สามารถจัดประเภท:", 
                 font=('Arial', 10, 'bold'), width=12, foreground='#e67e22').pack(side="left", padx=(0, 10))
        
        entry = ttk.Entry(unclass_frame, textvariable=self.unclassified_folder, width=50, font=('Arial', 9))
        entry.pack(side="left", fill="x", expand=True, padx=(0, 10))
        
        ttk.Button(unclass_frame, text="📁 เลือกโฟลเดอร์", 
                  command=self.select_unclassified_folder,
                  style='Custom.TButton').pack(side="right")

    def update_threshold_label(self, value):
        """Update threshold label"""
        self.threshold_label.configure(text=f"{float(value):.2f}")

    def select_prototype(self, idx):
        """Select prototype image for a class"""
        file = filedialog.askopenfilename(
            title=f"เลือกรูปตัวอย่างสำหรับประเภทที่ {idx+1}",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.gif"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("All files", "*.*")
            ]
        )
        if file:
            self.prototype_images[idx].set(file)
            self.status_var.set(f"เลือกรูปตัวอย่างประเภทที่ {idx+1} เรียบร้อย")

    def select_source_folder(self):
        """Select source folder"""
        folder = filedialog.askdirectory(title="เลือกโฟลเดอร์ที่มีรูปภาพต้นฉบับ")
        if folder:
            self.source_folder.set(folder)
            # Count images in folder
            try:
                image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')
                image_count = len([f for f in os.listdir(folder) 
                                 if f.lower().endswith(image_extensions)])
                self.status_var.set(f"เลือกโฟลเดอร์ต้นทาง: พบรูปภาพ {image_count} ไฟล์")
            except Exception as e:
                self.status_var.set(f"เลือกโฟลเดอร์ต้นทาง: {folder}")
                print(f"Error counting files: {e}")

    def select_destination(self, idx):
        """Select destination folder for a class"""
        folder = filedialog.askdirectory(title=f"เลือกโฟลเดอร์ปลายทางสำหรับประเภทที่ {idx+1}")
        if folder:
            self.destination_folders[idx].set(folder)
            self.status_var.set(f"เลือกโฟลเดอร์ปลายทางประเภทที่ {idx+1} เรียบร้อย")

    def select_unclassified_folder(self):
        """Select unclassified folder"""
        folder = filedialog.askdirectory(title="เลือกโฟลเดอร์สำหรับรูปที่ไม่สามารถจัดประเภทได้")
        if folder:
            self.unclassified_folder.set(folder)
            self.status_var.set("เลือกโฟลเดอร์รูปที่ไม่สามารถจัดประเภทได้เรียบร้อย")

    def extract_features(self, image_path):
        """Extract features from image using multiple methods"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None
                
            # Resize for consistency
            img = cv2.resize(img, (224, 224))
            
            # Convert color spaces
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Color histograms
            hist_b = cv2.calcHist([img], [0], None, [32], [0, 256])
            hist_g = cv2.calcHist([img], [1], None, [32], [0, 256])
            hist_r = cv2.calcHist([img], [2], None, [32], [0, 256])
            hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180])
            hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
            
            # Edge detection
            edges = cv2.Canny(gray, 50, 150)
            edge_hist = cv2.calcHist([edges], [0], None, [32], [0, 256])
            
            # Texture features
            texture = cv2.Laplacian(gray, cv2.CV_64F)
            texture_hist = cv2.calcHist([np.uint8(np.abs(texture))], [0], None, [32], [0, 256])
            
            # Combine all features
            features = np.concatenate([
                hist_b.flatten(), hist_g.flatten(), hist_r.flatten(),
                hist_h.flatten(), hist_s.flatten(),
                edge_hist.flatten(), texture_hist.flatten()
            ])
            
            # Normalize features
            norm = np.linalg.norm(features)
            if norm > 0:
                features = features / norm
            
            return features
            
        except Exception as e:
            print(f"Error extracting features from {image_path}: {e}")
            return None

    def validate_inputs(self):
        """Validate all inputs before starting classification"""
        errors = []
        
        # Check prototype images
        for i in range(self.num_classes.get()):
            proto_path = self.prototype_images[i].get()
            if not proto_path:
                errors.append(f"กรุณาเลือกรูปตัวอย่างสำหรับประเภทที่ {i+1}")
            elif not os.path.exists(proto_path):
                errors.append(f"ไม่พบไฟล์รูปตัวอย่างประเภทที่ {i+1}")
        
        # Check destination folders
        for i in range(self.num_classes.get()):
            dest_path = self.destination_folders[i].get()
            if not dest_path:
                errors.append(f"กรุณาเลือกโฟลเดอร์ปลายทางสำหรับประเภทที่ {i+1}")
        
        # Check source folder
        if not self.source_folder.get():
            errors.append("กรุณาเลือกโฟลเดอร์ต้นทาง")
        elif not os.path.exists(self.source_folder.get()):
            errors.append("ไม่พบโฟลเดอร์ต้นทาง")
        
        # Check unclassified folder
        if not self.unclassified_folder.get():
            errors.append("กรุณาเลือกโฟลเดอร์สำหรับรูปที่ไม่สามารถจัดประเภทได้")
        
        if errors:
            messagebox.showerror("ข้อมูลไม่ครบถ้วน", "\n".join(errors))
            return False
        
        # Create directories if they don't exist
        try:
            for i in range(self.num_classes.get()):
                os.makedirs(self.destination_folders[i].get(), exist_ok=True)
            os.makedirs(self.unclassified_folder.get(), exist_ok=True)
        except Exception as e:
            messagebox.showerror("ไม่สามารถสร้างโฟลเดอร์ได้", str(e))
            return False
        
        return True

    def start_classification(self):
        """Start the classification process"""
        if not self.validate_inputs():
            return
        
        if self.processing:
            messagebox.showwarning("กำลังทำงาน", "ระบบกำลังทำงานอยู่ กรุณารอสักครู่")
            return
        
        # Confirm start
        if not messagebox.askyesno("ยืนยันการทำงาน", 
                                  "คุณต้องการเริ่มจัดแยกรูปภาพหรือไม่?\n\nกระบวนการนี้อาจใช้เวลาสักครู่"):
            return
        
        self.processing = True
        self.stop_processing = False
        self.start_btn.configure(state='disabled')
        self.stop_btn.configure(state='normal')
        
        # Start classification in separate thread
        threading.Thread(target=self.classify_images, daemon=True).start()

    def stop_classification(self):
        """Stop the classification process"""
        self.stop_processing = True
        self.current_operation.set("กำลังหยุดการทำงาน...")
        self.status_var.set("กำลังหยุดการทำงาน กรุณารอสักครู่")

    def classify_images(self):
        """Main classification function"""
        try:
            self.current_operation.set("🔍 กำลังวิเคราะห์รูปตัวอย่าง...")
            self.progress_var.set(0)
            
            # Extract features from prototype images
            proto_features = {}
            for i in range(self.num_classes.get()):
                if self.stop_processing:
                    return
                    
                proto_path = self.prototype_images[i].get()
                self.current_operation.set(f"🔍 กำลังวิเคราะห์รูปตัวอย่างประเภทที่ {i+1}...")
                features = self.extract_features(proto_path)
                
                if features is None:
                    messagebox.showerror("ข้อผิดพลาด", 
                                       f"ไม่สามารถวิเคราะห์รูปตัวอย่างประเภทที่ {i+1} ได้")
                    return
                
                proto_features[i] = features
                self.progress_var.set(10 + (i * 10))
            
            # Get list of image files
            source_path = Path(self.source_folder.get())
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}
            image_files = [f for f in source_path.iterdir() 
                          if f.suffix.lower() in image_extensions and f.is_file()]
            
            if not image_files:
                messagebox.showwarning("ไม่พบรูปภาพ", "ไม่พบไฟล์รูปภาพในโฟลเดอร์ต้นทาง")
                return
            
            # Exclude prototype images from processing
            prototype_paths = {Path(p.get()).resolve() for p in self.prototype_images.values() if p.get()}
            image_files = [f for f in image_files if f.resolve() not in prototype_paths]
            
            total_files = len(image_files)
            processed = 0
            classified = {i: 0 for i in range(self.num_classes.get())}
            unclassified = 0
            
            self.current_operation.set(f"📊 พบรูปภาพทั้งหมด {total_files} ไฟล์")
            
            # Process each image
            for idx, image_file in enumerate(image_files):
                if self.stop_processing:
                    break
                
                self.current_operation.set(f"🔄 กำลังประมวลผล: {image_file.name} ({idx+1}/{total_files})")
                
                # Extract features
                features = self.extract_features(str(image_file))
                if features is None:
                    continue
                
                # Calculate similarity scores
                scores = {}
                for i in proto_features:
                    similarity = cosine_similarity([features], [proto_features[i]])[0][0]
                    scores[i] = similarity
                
                # Find best match
                best_class = max(scores, key=scores.get)
                confidence = scores[best_class]
                
                # Get current threshold
                current_threshold = self.threshold.get()
                
                # Classify based on confidence threshold
                if confidence > current_threshold:
                    dest_folder = Path(self.destination_folders[best_class].get())
                    classified[best_class] += 1
                    classification_info = f"ประเภทที่ {best_class+1} (ความแม่น: {confidence:.3f})"
                else:
                    dest_folder = Path(self.unclassified_folder.get())
                    unclassified += 1
                    classification_info = f"ไม่สามารถจำแนกได้ (ความแม่นสูงสุด: {confidence:.3f})"
                
                # Copy file with unique name
                dest_path = dest_folder / image_file.name
                counter = 1
                while dest_path.exists():
                    stem = image_file.stem
                    suffix = image_file.suffix
                    dest_path = dest_folder / f"{stem}_{counter}{suffix}"
                    counter += 1
                
                try:
                    shutil.copy2(image_file, dest_path)
                    processed += 1
                except Exception as e:
                    print(f"Error copying {image_file}: {e}")
                
                # Update progress
                progress = 20 + (idx * 80 / total_files)
                self.progress_var.set(progress)
                
                # Update status
                status_text = f"ประมวลผลแล้ว: {processed}/{total_files} | {classification_info}"
                self.status_var.set(status_text)
            
            # Final results
            if not self.stop_processing:
                self.progress_var.set(100)
                self.current_operation.set("✅ เสร็จสิ้น!")
                
                # Create summary
                summary = [f"📊 สรุปผลการจัดแยกรูปภาพ"]
                summary.append(f"รูปภาพทั้งหมด: {total_files} ไฟล์")
                summary.append(f"ประมวลผลสำเร็จ: {processed} ไฟล์")
                summary.append("")
                
                for i in range(self.num_classes.get()):
                    summary.append(f"ประเภทที่ {i+1}: {classified[i]} ไฟล์")
                summary.append(f"ไม่สามารถจำแนกได้: {unclassified} ไฟล์")
                
                self.status_var.set("เสร็จสิ้นการจัดแยกรูปภาพแล้ว")
                messagebox.showinfo("เสร็จสิ้น", "\n".join(summary))
            else:
                self.current_operation.set("⏹️ หยุดการทำงานแล้ว")
                self.status_var.set(f"หยุดการทำงาน - ประมวลผลแล้ว {processed}/{total_files} ไฟล์")
                
        except Exception as e:
            self.current_operation.set("❌ เกิดข้อผิดพลาด")
            self.status_var.set(f"เกิดข้อผิดพลาด: {str(e)}")
            messagebox.showerror("ข้อผิดพลาด", f"เกิดข้อผิดพลาดระหว่างการทำงาน:\n{str(e)}")
        finally:
            # Reset UI state
            self.processing = False
            self.stop_processing = False
            self.start_btn.configure(state='normal')
            self.stop_btn.configure(state='disabled')

    def reset_all(self):
        """Reset all inputs and UI"""
        if self.processing:
            if not messagebox.askyesno("ยืนยันการรีเซ็ต", 
                                      "ระบบกำลังทำงานอยู่ การรีเซ็ตจะหยุดการทำงาน\nคุณต้องการดำเนินการต่อหรือไม่?"):
                return
            self.stop_processing = True
        
        # Reset all variables
        self.num_classes.set(2)
        self.source_folder.set("")
        self.unclassified_folder.set("")
        self.progress_var.set(0)
        self.status_var.set("พร้อมใช้งาน")
        self.current_operation.set("")
        
        # Update UI
        self.update_class_inputs()
        
        # Reset button states
        self.start_btn.configure(state='normal')
        self.stop_btn.configure(state='disabled')
        self.processing = False
        self.stop_processing = False

    def on_closing(self):
        """Handle window closing event"""
        if self.processing:
            if messagebox.askyesno("ยืนยันการปิดโปรแกรม", 
                                  "ระบบกำลังทำงานอยู่ การปิดโปรแกรมจะหยุดการทำงาน\nคุณต้องการปิดโปรแกรมหรือไม่?"):
                self.stop_processing = True
                self.root.destroy()
        else:
            self.root.destroy()

def main():
    """Main function to run the application"""
    root = tk.Tk()
    
    # Set window icon (if available)
    try:
        root.iconbitmap('icon.ico')  # Add icon file if available
    except:
        pass
    
    # Create application instance
    app = ImageSimilarityClassifier(root)
    
    # Handle window closing
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    # Center window on screen
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')
    
    # Start the application
    root.mainloop()

if __name__ == "__main__":
    main()