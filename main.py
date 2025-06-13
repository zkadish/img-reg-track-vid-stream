import cv2
import numpy as np
import time
from pathlib import Path
from src.detector import ObjectDetector
from utils.video_utils import get_video_source, read_frame
from utils.preprocessing import preprocess_frame
from config.settings import VIDEO_SOURCE, PREPROCESSING

def main():
    try:
        # Initialize video capture
        cap = get_video_source(VIDEO_SOURCE)
        
        # Initialize YOLO detector
        detector = ObjectDetector(confidence=0.5)
        
        print("Press 'q' to quit")
        
        # FPS calculation variables
        frame_count = 0
        start_time = time.time()
        fps = 0
        
        while True:
            # Read frame from video stream
            ret, frame = read_frame(cap)
            
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Calculate FPS
            frame_count += 1
            if frame_count >= 30:  # Calculate FPS every 30 frames
                end_time = time.time()
                fps = frame_count / (end_time - start_time)
                print(f"Current FPS: {fps:.2f}")
                frame_count = 0
                start_time = time.time()
            
            # Print frame dimensions
            height, width = frame.shape[:2]
            print(f"Output frame size: {width}x{height}")
            
            # Apply preprocessing
            processed_frame = preprocess_frame(
                frame,
                target_size=PREPROCESSING["target_size"],
                normalize=PREPROCESSING["normalize"],
                denoise=PREPROCESSING["denoise"],
                enhance_contrast=PREPROCESSING["enhance_contrast"]
            )
            
            # Detect objects
            detections = detector.detect(processed_frame)
            
            # Draw detections on original frame
            frame = detector.draw_detections(frame, detections)
            
            # Add FPS text to frame
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display the frame
            cv2.imshow('Object Detection', frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except Exception as e:
        print(f"Error: {str(e)}")
    
    finally:
        # Clean up
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 