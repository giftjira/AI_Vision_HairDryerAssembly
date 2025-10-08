import cv2
import numpy as np
import os
from PIL import Image, ImageEnhance, ImageFilter
import random
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter import *
import threading
import math

class DataAugmentationTool:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Data Augmentation Tool")
        self.root.geometry("900x700")
        self.root.configure(bg='#f0f0f0')
        
        self.input_path = ""
        self.output_path = ""
        self.selected_methods = []
        self.total_input_images = 0
        
        self.setup_ui()
    
    def setup_ui(self):
        # Title
        title_label = tk.Label(self.root, text="Data Augmentation Tool", 
                              font=('Arial', 20, 'bold'), bg='#f0f0f0', fg='#333')
        title_label.pack(pady=20)
        
        # Input selection frame
        input_frame = tk.Frame(self.root, bg='#f0f0f0')
        input_frame.pack(pady=10, padx=20, fill='x')
        
        tk.Label(input_frame, text="Input Folder:", font=('Arial', 12), 
                bg='#f0f0f0').pack(anchor='w')
        
        input_path_frame = tk.Frame(input_frame, bg='#f0f0f0')
        input_path_frame.pack(fill='x', pady=5)
        
        self.input_label = tk.Label(input_path_frame, text="No folder selected", 
                                   bg='white', relief='sunken', anchor='w')
        self.input_label.pack(side='left', fill='x', expand=True, padx=(0, 10))
        
        tk.Button(input_path_frame, text="Browse", command=self.select_input_folder,
                 bg='#4CAF50', fg='white', font=('Arial', 10)).pack(side='right')
        
        # Image count display
        self.image_count_label = tk.Label(input_frame, text="Images found: 0", 
                                         font=('Arial', 10), bg='#f0f0f0', fg='#666')
        self.image_count_label.pack(anchor='w', pady=(5, 0))
        
        # Output selection frame
        output_frame = tk.Frame(self.root, bg='#f0f0f0')
        output_frame.pack(pady=10, padx=20, fill='x')
        
        tk.Label(output_frame, text="Output Folder:", font=('Arial', 12), 
                bg='#f0f0f0').pack(anchor='w')
        
        output_path_frame = tk.Frame(output_frame, bg='#f0f0f0')
        output_path_frame.pack(fill='x', pady=5)
        
        self.output_label = tk.Label(output_path_frame, text="No folder selected", 
                                    bg='white', relief='sunken', anchor='w')
        self.output_label.pack(side='left', fill='x', expand=True, padx=(0, 10))
        
        tk.Button(output_path_frame, text="Browse", command=self.select_output_folder,
                 bg='#4CAF50', fg='white', font=('Arial', 10)).pack(side='right')
        
        # Number of images to generate frame
        generate_frame = tk.LabelFrame(self.root, text="Number of Images to Generate", 
                                      font=('Arial', 12, 'bold'), bg='#f0f0f0')
        generate_frame.pack(pady=10, padx=20, fill='x')
        
        # Spinbox for selecting number of images
        number_frame = tk.Frame(generate_frame, bg='#f0f0f0')
        number_frame.pack(pady=10, padx=10, fill='x')
        
        tk.Label(number_frame, text="Total images to generate:", 
                font=('Arial', 11), bg='#f0f0f0').pack(side='left')
        
        self.num_images_var = tk.IntVar(value=100)
        self.num_images_spinbox = tk.Spinbox(number_frame, from_=1, to=10000, 
                                           textvariable=self.num_images_var,
                                           font=('Arial', 11), width=10)
        self.num_images_spinbox.pack(side='left', padx=(10, 0))
        
        # Info label
        self.generation_info_label = tk.Label(generate_frame, 
                                            text="Select augmentation methods below, then click 'Generate Images'",
                                            font=('Arial', 10), bg='#f0f0f0', fg='#666')
        self.generation_info_label.pack(pady=(5, 10))
        
        # Augmentation methods frame
        methods_frame = tk.LabelFrame(self.root, text="Augmentation Methods", 
                                     font=('Arial', 12, 'bold'), bg='#f0f0f0')
        methods_frame.pack(pady=10, padx=20, fill='both', expand=True)
        
        # Create scrollable frame for methods
        canvas = tk.Canvas(methods_frame, bg='#f0f0f0', height=200)
        scrollbar = ttk.Scrollbar(methods_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='#f0f0f0')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Augmentation methods
        self.method_vars = {}
        methods = {
            "Geometric Transformations": [
                ("rotation", "Rotation (0-360°)"),
                ("scaling", "Scaling (0.8-1.2x)"),
                ("flipping", "Horizontal/Vertical Flip"),
                ("cropping", "Random Cropping")
            ],
            "Color Adjustments": [
                ("brightness", "Brightness Adjustment"),
                ("contrast", "Contrast Adjustment"),
                ("saturation", "Saturation Adjustment"),
                ("noise", "Add Random Noise")
            ],
            "Advanced Augmentations": [
                ("distortion", "Perspective Distortion"),
                ("blur", "Gaussian Blur")
            ]
        }
        
        for category, method_list in methods.items():
            # Category label
            cat_label = tk.Label(scrollable_frame, text=category, 
                               font=('Arial', 11, 'bold'), bg='#f0f0f0', fg='#2196F3')
            cat_label.pack(anchor='w', pady=(10, 5))
            
            # Methods in category
            for method_key, method_name in method_list:
                var = tk.BooleanVar()
                self.method_vars[method_key] = var
                cb = tk.Checkbutton(scrollable_frame, text=method_name, variable=var,
                                   bg='#f0f0f0', font=('Arial', 10))
                cb.pack(anchor='w', padx=20)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Bind mousewheel to canvas
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Main Generate Button (Large and prominent)
        generate_main_frame = tk.Frame(self.root, bg='#f0f0f0')
        generate_main_frame.pack(pady=15)
        
        self.generate_button = tk.Button(generate_main_frame, text="🚀 START GENERATING IMAGES 🚀", 
                                       command=self.start_augmentation,
                                       bg='#FF5722', fg='white', 
                                       font=('Arial', 14, 'bold'), 
                                       padx=30, pady=12,
                                       relief='raised', bd=3,
                                       cursor='hand2')
        self.generate_button.pack()
        
        # Secondary control buttons frame
        button_frame = tk.Frame(self.root, bg='#f0f0f0')
        button_frame.pack(pady=10)
        
        tk.Button(button_frame, text="Select All Methods", command=self.select_all,
                 bg='#4CAF50', fg='white', font=('Arial', 10), 
                 padx=15, pady=8).pack(side='left', padx=5)
        
        tk.Button(button_frame, text="Clear All", command=self.clear_selection,
                 bg='#607D8B', fg='white', font=('Arial', 10), 
                 padx=15, pady=8).pack(side='left', padx=5)
        
        tk.Button(button_frame, text="Reset Settings", command=self.reset_all,
                 bg='#9E9E9E', fg='white', font=('Arial', 10), 
                 padx=15, pady=8).pack(side='left', padx=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(self.root, mode='determinate')
        self.progress.pack(pady=10, padx=20, fill='x')
        
        # Status label
        self.status_label = tk.Label(self.root, text="Ready - Select input folder to begin", 
                                    bg='#f0f0f0', font=('Arial', 10), fg='#666')
        self.status_label.pack(pady=5)
    
    def select_input_folder(self):
        folder = filedialog.askdirectory(title="Select Input Folder")
        if folder:
            self.input_path = folder
            self.input_label.config(text=folder)
            
            # Count images in the folder
            try:
                image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')
                image_files = []
                
                for f in os.listdir(folder):
                    if f.lower().endswith(image_extensions):
                        # Check if file is actually readable
                        try:
                            img_path = os.path.join(folder, f)
                            test_img = cv2.imread(img_path)
                            if test_img is not None:
                                image_files.append(f)
                        except:
                            continue
                
                self.total_input_images = len(image_files)
                self.image_count_label.config(text=f"Images found: {self.total_input_images}")
                
                if self.total_input_images == 0:
                    messagebox.showwarning("Warning", "No valid image files found in the selected folder!")
                    self.status_label.config(text="No images found in input folder")
                else:
                    self.status_label.config(text=f"Ready - {self.total_input_images} images found")
                    
            except Exception as e:
                messagebox.showerror("Error", f"Error reading folder: {str(e)}")
                self.image_count_label.config(text="Images found: Error")
    
    def select_output_folder(self):
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            self.output_path = folder
            self.output_label.config(text=folder)
            
            # Create output folder if it doesn't exist
            try:
                os.makedirs(folder, exist_ok=True)
            except Exception as e:
                messagebox.showerror("Error", f"Cannot create output folder: {str(e)}")
    
    def clear_selection(self):
        for var in self.method_vars.values():
            var.set(False)
    
    def select_all(self):
        for var in self.method_vars.values():
            var.set(True)
    
    def reset_all(self):
        """Reset all settings to default"""
        self.clear_selection()
        self.num_images_var.set(100)
        self.input_path = ""
        self.output_path = ""
        self.total_input_images = 0
        self.input_label.config(text="No folder selected")
        self.output_label.config(text="No folder selected")
        self.image_count_label.config(text="Images found: 0")
        self.status_label.config(text="Ready - Select input folder to begin")
        self.progress['value'] = 0
    
    def start_augmentation(self):
        # Validation
        if not self.input_path or not self.output_path:
            messagebox.showerror("Error", "Please select input and output folders!")
            return
        
        if self.total_input_images == 0:
            messagebox.showerror("Error", "No images found in input folder!")
            return
        
        selected_methods = [method for method, var in self.method_vars.items() if var.get()]
        if not selected_methods:
            messagebox.showerror("Error", "Please select at least one augmentation method!")
            return
        
        target_images = self.num_images_var.get()
        if target_images <= 0:
            messagebox.showerror("Error", "Please enter a valid number of images to generate!")
            return
        
        # Confirm generation
        result = messagebox.askyesno("Confirm Generation", 
            f"Generate {target_images} images using:\n"
            f"• Input images: {self.total_input_images}\n"
            f"• Augmentation methods: {len(selected_methods)}\n"
            f"• Output folder: {os.path.basename(self.output_path)}\n\n"
            f"Continue?")
        
        if not result:
            return
        
        # Disable generate button during processing
        self.generate_button.config(state='disabled', text='⏳ GENERATING... PLEASE WAIT ⏳',
                                  bg='#9E9E9E')
        
        # Start augmentation in separate thread
        thread = threading.Thread(target=self.perform_augmentation, 
                                args=(selected_methods, target_images))
        thread.daemon = True
        thread.start()
    
    def perform_augmentation(self, methods, target_count):
        try:
            # Get image files
            image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif')
            image_files = []
            
            for f in os.listdir(self.input_path):
                if f.lower().endswith(image_extensions):
                    try:
                        # Test if image can be read
                        img_path = os.path.join(self.input_path, f)
                        test_img = cv2.imread(img_path)
                        if test_img is not None:
                            image_files.append(f)
                    except:
                        continue
            
            if not image_files:
                raise Exception("No valid image files found!")
            
            # Setup progress
            self.progress['maximum'] = target_count
            self.progress['value'] = 0
            
            generated_count = 0
            
            # Calculate how many augmentations we need
            total_possible = len(image_files) * (len(methods) + 1)  # +1 for original
            actual_target = min(target_count, total_possible)
            
            # Generate images
            while generated_count < actual_target:
                for filename in image_files:
                    if generated_count >= actual_target:
                        break
                    
                    try:
                        input_file = os.path.join(self.input_path, filename)
                        base_name = os.path.splitext(filename)[0]
                        
                        # Load image
                        image = cv2.imread(input_file)
                        if image is None:
                            continue
                        
                        # Save original image first
                        if generated_count < actual_target:
                            output_file = os.path.join(self.output_path, f"{base_name}_original_{generated_count:04d}.jpg")
                            cv2.imwrite(output_file, image)
                            generated_count += 1
                            
                            self.progress['value'] = generated_count
                            self.status_label.config(text=f"Generated {generated_count}/{actual_target} images")
                            self.root.update_idletasks()
                        
                        # Generate augmented versions
                        for method in methods:
                            if generated_count >= actual_target:
                                break
                            
                            try:
                                aug_image = self.apply_single_augmentation(image.copy(), method)
                                output_file = os.path.join(self.output_path, 
                                                         f"{base_name}_{method}_{generated_count:04d}.jpg")
                                
                                # Ensure the augmented image is valid
                                if aug_image is not None and aug_image.size > 0:
                                    success = cv2.imwrite(output_file, aug_image)
                                    if success:
                                        generated_count += 1
                                        self.progress['value'] = generated_count
                                        self.status_label.config(text=f"Generated {generated_count}/{actual_target} images")
                                        self.root.update_idletasks()
                                    
                            except Exception as e:
                                print(f"Error applying {method} to {filename}: {e}")
                                continue
                                
                    except Exception as e:
                        print(f"Error processing {filename}: {e}")
                        continue
                
                # If we've gone through all images and methods but haven't reached target,
                # we'll need to repeat with different random parameters
                if generated_count < actual_target and generated_count > 0:
                    # We've generated what we can with current approach
                    break
            
            # Complete
            self.status_label.config(text=f"Completed! Generated {generated_count} images")
            messagebox.showinfo("Success", 
                f"Augmentation completed!\n"
                f"Generated {generated_count} images\n"
                f"Saved to: {self.output_path}")
            
        except Exception as e:
            self.status_label.config(text="Error occurred")
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            print(f"Full error: {e}")
        
        finally:
            # Re-enable generate button
            self.generate_button.config(state='normal', text='🚀 START GENERATING IMAGES 🚀',
                                      bg='#FF5722')
    
    def apply_single_augmentation(self, image, method):
        """Apply a single augmentation method to an image"""
        try:
            if image is None or image.size == 0:
                return image
                
            h, w = image.shape[:2]
            
            if method == "rotation":
                angle = random.uniform(-30, 30)
                center = (w//2, h//2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            
            elif method == "scaling":
                scale = random.uniform(0.8, 1.2)
                new_w, new_h = int(w * scale), int(h * scale)
                scaled = cv2.resize(image, (new_w, new_h))
                
                if scale > 1:
                    # Crop to original size
                    y_start = max(0, (new_h - h) // 2)
                    x_start = max(0, (new_w - w) // 2)
                    return scaled[y_start:y_start+h, x_start:x_start+w]
                else:
                    # Pad to original size
                    pad_h = (h - new_h) // 2
                    pad_w = (w - new_w) // 2
                    return cv2.copyMakeBorder(scaled, pad_h, h-new_h-pad_h, 
                                            pad_w, w-new_w-pad_w, cv2.BORDER_REFLECT)
            
            elif method == "flipping":
                flip_code = random.choice([0, 1])  # 0=vertical, 1=horizontal
                return cv2.flip(image, flip_code)
            
            elif method == "cropping":
                crop_ratio = random.uniform(0.7, 0.9)
                crop_h = int(h * crop_ratio)
                crop_w = int(w * crop_ratio)
                
                y_start = random.randint(0, max(1, h - crop_h))
                x_start = random.randint(0, max(1, w - crop_w))
                
                cropped = image[y_start:y_start+crop_h, x_start:x_start+crop_w]
                return cv2.resize(cropped, (w, h))
            
            elif method == "brightness":
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                hsv = hsv.astype(np.float32)
                brightness_factor = random.uniform(0.7, 1.3)
                hsv[:, :, 2] *= brightness_factor
                hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
                hsv = hsv.astype(np.uint8)
                return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
            elif method == "contrast":
                contrast_factor = random.uniform(0.7, 1.3)
                return cv2.convertScaleAbs(image, alpha=contrast_factor, beta=0)
            
            elif method == "saturation":
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                hsv = hsv.astype(np.float32)
                saturation_factor = random.uniform(0.7, 1.3)
                hsv[:, :, 1] *= saturation_factor
                hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
                hsv = hsv.astype(np.uint8)
                return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
            elif method == "noise":
                noise_factor = random.uniform(10, 30)
                noise = np.random.normal(0, noise_factor, image.shape).astype(np.float32)
                noisy = image.astype(np.float32) + noise
                return np.clip(noisy, 0, 255).astype(np.uint8)
            
            elif method == "blur":
                kernel_size = random.choice([3, 5, 7])
                return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
            
            elif method == "distortion":
                # Simple perspective distortion
                pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
                offset = min(w, h) // 20  # Dynamic offset based on image size
                pts2 = np.float32([
                    [random.randint(0, offset), random.randint(0, offset)], 
                    [w-random.randint(0, offset), random.randint(0, offset)], 
                    [random.randint(0, offset), h-random.randint(0, offset)], 
                    [w-random.randint(0, offset), h-random.randint(0, offset)]
                ])
                M = cv2.getPerspectiveTransform(pts1, pts2)
                return cv2.warpPerspective(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            
            return image
            
        except Exception as e:
            print(f"Error in augmentation {method}: {e}")
            return image
    
    def run(self):
        self.root.mainloop()

# Usage
if __name__ == "__main__":
    try:
        app = DataAugmentationTool()
        app.run()
    except Exception as e:
        print(f"Error starting application: {e}")
        input("Press Enter to exit...")