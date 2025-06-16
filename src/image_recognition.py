import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from config.settings import YOLO_MODEL_PATH, YOLO_CONFIDENCE_THRESHOLD

class ObjectImageRecognition:
    def __init__(self, confidence=YOLO_CONFIDENCE_THRESHOLD):
        """Initialize the YOLO object recognition model.
        
        Args:
            confidence (float): Confidence threshold for detections
        """
        self.confidence = confidence
        self.model = self._load_model()
        self.class_names = self.model.names
        
    def _load_model(self):
        """Load the YOLO model."""
        try:
            # Create models directory if it doesn't exist
            Path('models').mkdir(exist_ok=True)
            
            # Download model if it doesn't exist
            if not Path(YOLO_MODEL_PATH).exists():
                print("Downloading YOLO model...")
                model = YOLO('yolov8n.pt')  # Download and load the model
                model.save(YOLO_MODEL_PATH)  # Save it to our models directory
                print("Model downloaded successfully")
            else:
                print("Loading existing YOLO model...")
                model = YOLO(YOLO_MODEL_PATH)
            
            print("Model loaded successfully")
            return model
            
        except Exception as e:
            print(f"Error loading YOLO model: {str(e)}")
            raise
            
    def detect(self, frame):
        """Perform object detection on a frame.
        
        Args:
            frame (numpy.ndarray): Input frame
            
        Returns:
            list: List of detections as (bbox, confidence, class_id) tuples
        """
        try:
            # Perform detection
            results = self.model(frame, conf=self.confidence)[0]
            
            # Process results
            detections = []
            for r in results.boxes.data.tolist():
                x1, y1, x2, y2, confidence, class_id = r
                bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                detections.append((bbox, confidence, int(class_id)))
                    
            return detections
            
        except Exception as e:
            print(f"Error during detection: {str(e)}")
            return []
            
    def draw_detections(self, frame, detections):
        """Draw detection boxes and labels on frame.
        
        Args:
            frame (numpy.ndarray): Input frame
            detections (list): List of detections from detect()
            
        Returns:
            numpy.ndarray: Frame with detections drawn
        """
        try:
            for bbox, conf, class_id in detections:
                x, y, w, h = bbox
                
                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Draw label
                label = f"{self.class_names[class_id]}: {conf:.2f}"
                cv2.putText(frame, label, (x, y - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                          
            return frame
            
        except Exception as e:
            print(f"Error drawing detections: {str(e)}")
            return frame 