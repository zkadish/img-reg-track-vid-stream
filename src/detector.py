from ultralytics import YOLO
import cv2
import numpy as np
from typing import List, Tuple, Optional
import os
import torch
import urllib.request
import ssl

class ObjectDetector:
    def __init__(self, model_path: str = "yolov8n.pt", confidence: float = 0.5):
        """
        Initialize the YOLO detector
        Args:
            model_path: Path to the YOLO model weights
            confidence: Detection confidence threshold
        """
        try:
            # Create models directory if it doesn't exist
            os.makedirs('models', exist_ok=True)
            model_path = os.path.join('models', 'yolov8n.pt')
            
            # Download model if it doesn't exist
            if not os.path.exists(model_path):
                print("Downloading YOLO model...")
                # Create an unverified SSL context
                ssl._create_default_https_context = ssl._create_unverified_context
                # Download the model
                url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
                urllib.request.urlretrieve(url, model_path)
                print("Model downloaded successfully")
            
            # Load the model
            print("Loading YOLO model...")
            self.model = YOLO(model_path)
            print("Model loaded successfully")
            
            # Verify model is loaded
            if not hasattr(self.model, 'model'):
                raise Exception("Model not loaded properly")
                
            self.confidence = confidence
            self.class_names = self.model.names
            print(f"Model loaded successfully. Classes: {self.class_names}")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

    def detect(self, frame: np.ndarray) -> List[Tuple]:
        """
        Detect objects in the frame
        Args:
            frame: Input frame
        Returns:
            List of detections as (bbox, confidence, class_id) tuples
        """
        try:
            # Convert frame to RGB (YOLO expects RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run inference
            results = self.model(frame_rgb, conf=self.confidence)[0]
            detections = []
            
            for r in results.boxes.data.tolist():
                x1, y1, x2, y2, confidence, class_id = r
                bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                detections.append((bbox, confidence, int(class_id)))
                
            return detections
        except Exception as e:
            print(f"Error during detection: {str(e)}")
            return []

    def draw_detections(self, frame: np.ndarray, detections: List[Tuple]) -> np.ndarray:
        """
        Draw bounding boxes and labels on the frame
        Args:
            frame: Input frame
            detections: List of detections
        Returns:
            Frame with drawn detections
        """
        for bbox, confidence, class_id in detections:
            x, y, w, h = bbox
            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Prepare label
            label = f"{self.class_names[class_id]}: {confidence:.2f}"
            
            # Draw label background
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x, y - label_h - 10), (x + label_w, y), (0, 255, 0), -1)
            
            # Draw label text
            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
        return frame 