import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO as YOLOModel
from config.settings import YOLO as YOLO_CONFIG

class ObjectImageRecognition:
    def __init__(self, confidence=0.5):
        """Initialize YOLO model.
        
        Args:
            confidence (float): Minimum confidence for detections
        """
        self.model = YOLOModel(YOLO_CONFIG["model_path"])
        self.confidence = confidence
        self.class_names = self.model.names
        
    def _load_model(self):
        """Load the YOLO model."""
        try:
            # Create models directory if it doesn't exist
            Path('models').mkdir(exist_ok=True)
            
            # Download model if it doesn't exist
            if not Path(YOLO_CONFIG["model_path"]).exists():
                print("Downloading YOLO model...")
                model = YOLOModel('yolov8n.pt')  # Download and load the model
                model.save(YOLO_CONFIG["model_path"])  # Save it to our models directory
                print("Model downloaded successfully")
            else:
                print("Loading existing YOLO model...")
                model = YOLOModel(YOLO_CONFIG["model_path"])
            
            print("Model loaded successfully")
            return model
            
        except Exception as e:
            print(f"Error loading YOLO model: {str(e)}")
            raise
            
    def detect(self, frame):
        """Detect objects in frame.
        
        Args:
            frame (numpy.ndarray): Input frame
            
        Returns:
            list: List of (bbox, confidence, class_id) tuples
        """
        try:
            # Run YOLO detection
            results = self.model(frame, conf=self.confidence)[0]
            
            # Extract detections
            detections = []
            for r in results.boxes.data.tolist():
                x1, y1, x2, y2, conf, class_id = r
                bbox = [x1, y1, x2 - x1, y2 - y1]  # Convert to [x, y, w, h]
                detections.append((bbox, conf, int(class_id)))
                    
            return detections
            
        except Exception as e:
            print(f"Error detecting objects: {str(e)}")
            return []
            
    def draw_detections(self, frame, detections, box_color=(0, 255, 0), text_color=(0, 255, 0), 
                       box_thickness=2, text_scale=0.9, text_thickness=2):
        """Draw detections on frame.
        
        Args:
            frame (numpy.ndarray): Input frame
            detections (list): List of (bbox, confidence, class_id) tuples
            box_color (tuple): BGR color for detection boxes
            text_color (tuple): BGR color for text
            box_thickness (int): Line thickness for detection boxes
            text_scale (float): Text size
            text_thickness (int): Text thickness
            
        Returns:
            numpy.ndarray: Frame with detections drawn
        """
        try:
            for bbox, conf, class_id in detections:
                # Draw bounding box
                x, y, w, h = [int(v) for v in bbox]
                cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, box_thickness)
                
                # Draw label
                label = f"{self.class_names[class_id]}: {conf:.2f}"
                cv2.putText(
                    frame, 
                    label, 
                    (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    text_scale,
                    text_color,
                    text_thickness
                )
                          
            return frame
            
        except Exception as e:
            print(f"Error drawing detections: {str(e)}")
            return frame 