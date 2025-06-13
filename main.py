import cv2
import numpy as np
from pathlib import Path
from src.detector import ObjectDetector
from utils.video_utils import get_video_source, read_frame, resize_frame
from config.settings import VIDEO_SOURCE, FRAME_WIDTH, FRAME_HEIGHT

def main():
    # Initialize video capture
    cap = get_video_source(VIDEO_SOURCE)
    
    # Initialize YOLO detector
    detector = ObjectDetector(confidence=0.5)
    
    print("Press 'q' to quit")
    
    while True:
        # Read frame from video stream
        ret, frame = read_frame(cap)
        
        if not ret:
            print("Error: Could not read frame")
            break
            
        # Resize frame if needed
        frame = resize_frame(frame, FRAME_WIDTH, FRAME_HEIGHT)
        
        # Detect objects
        detections = detector.detect(frame)
        
        # Draw detections
        frame = detector.draw_detections(frame, detections)
        
        # Display the frame
        cv2.imshow('Object Detection', frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 