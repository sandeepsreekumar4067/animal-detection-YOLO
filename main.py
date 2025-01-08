import tkinter as tk
from tkinter import Label, Frame
import cv2
import torch
from PIL import Image, ImageTk

# Load YOLO model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Define animal classes
animal_classes = ['dog', 'cat', 'bird', 'horse', 'sheep', 'cow' , 'elephant' ,'tiger','bear']

# Tkinter GUI class
class AnimalDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Animal Detection with YOLO")
        
        # Video frame
        self.video_frame = Frame(root, width=640, height=480)
        self.video_frame.pack(side=tk.LEFT)
        
        # Sidebar
        self.sidebar = Frame(root, width=200, height=480, bg='white')
        self.sidebar.pack(side=tk.RIGHT, fill=tk.Y)
        Label(self.sidebar, text="Detected Animals", bg="white", font=("Arial", 16)).pack(pady=10)
        self.animal_list = tk.Listbox(self.sidebar, font=("Arial", 14), width=20, height=20)
        self.animal_list.pack(pady=10)
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(0)  # Use 0 for the default webcam
        
        # Placeholder for video display
        self.video_label = Label(self.video_frame)
        self.video_label.pack()
        
        # Start video stream
        self.update_frame()
    
    def detect_animals(self, frame):
        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Perform inference with YOLO
        results = model(rgb_frame)
        detections = results.pandas().xyxy[0]  # Get results as DataFrame
        
        # Filter animal detections
        animal_detections = detections[detections['name'].isin(animal_classes)]
        
        # Clear the listbox and update detected animals
        self.animal_list.delete(0, tk.END)
        detected_names = set()  # Avoid duplicate names
        for _, row in animal_detections.iterrows():
            detected_names.add(row['name'])
        
        for name in detected_names:
            self.animal_list.insert(tk.END, name)
        
        # Draw bounding boxes on the frame
        for _, row in animal_detections.iterrows():
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            label = row['name']
            confidence = row['confidence']
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label
            text = f"{label} {confidence:.2f}"
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame
    
    def update_frame(self):
        # Read frame from webcam
        ret, frame = self.cap.read()
        if not ret:
            return
        
        # Detect animals and update GUI
        frame = self.detect_animals(frame)
        
        # Convert frame to ImageTk format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        
        # Update video label
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)
        
        # Schedule the next update
        self.root.after(10, self.update_frame)
    
    def on_close(self):
        # Release the webcam and destroy the window
        self.cap.release()
        self.root.destroy()

# Initialize Tkinter application
root = tk.Tk()
app = AnimalDetectionApp(root)
root.protocol("WM_DELETE_WINDOW", app.on_close)
root.mainloop()
