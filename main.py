import cv2
import numpy as np
import time
from pathlib import Path
from src.detector import ObjectDetector
from src.tracker import ImageTracker
from utils.video_utils import get_video_source, read_frame
from utils.preprocessing import preprocess_frame
from config.settings import VIDEO_SOURCE, PREPROCESSING, TRACKER_TYPE

def main():
    try:
        # Initialize video capture
        cap = get_video_source(VIDEO_SOURCE)
        
        # Initialize YOLO detector and tracker
        print("Initializing YOLO detector...")
        detector = ObjectDetector(confidence=0.5)
        print("Initializing tracker...")
        tracker = ImageTracker(tracker_type=TRACKER_TYPE)
        
        print(f"Using {TRACKER_TYPE} tracker")
        print("Press 'q' to quit")
        print("Press 'r' to reset tracking")
        
        # FPS calculation variables
        frame_count = 0
        start_time = time.time()
        fps = 0
        
        # Tracking state
        tracking = False
        
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
            
            if not tracking:
                # Detect objects
                detections = detector.detect(processed_frame)
                
                # Draw detections on original frame
                frame = detector.draw_detections(frame, detections)
                
                # If we have detections, start tracking the first one
                if detections:
                    bbox, confidence, class_id = detections[0]
                    tracking = tracker.initialize(frame, bbox)
                    print(f"Started tracking with {TRACKER_TYPE} tracker")
            else:
                # Update tracker
                success, bbox = tracker.update(frame)
                if success:
                    frame = tracker.draw_bbox(frame)
                else:
                    print("Lost tracking, switching back to detection")
                    tracking = False
            
            # Add FPS text to frame
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display the frame
            cv2.imshow('Object Detection', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                tracking = False
                print("Reset tracking")
    
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Stack trace:")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 