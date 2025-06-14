import cv2
import numpy as np
import time
from pathlib import Path
from src.detector import ObjectDetector
from src.tracker import ImageTracker
from src.ui import TrackingUI
from src.motion_detector import MotionDetector
from utils.video_utils import get_video_source, read_frame
from utils.preprocessing import preprocess_frame
from config.settings import VIDEO_SOURCE, PREPROCESSING, TRACKER_TYPE

def main():
    try:
        # Initialize video capture
        cap = get_video_source(VIDEO_SOURCE)
        
        # Initialize components
        print("Initializing motion detector...")
        motion_detector = MotionDetector(min_area=100, history=5)
        
        # Initialize UI
        ui = TrackingUI()
        
        print("Press 'q' to quit")
        
        # FPS calculation variables
        frame_count = 0
        start_time = time.time()
        fps = 0
        
        # Wait for first frame
        ret, frame = read_frame(cap)
        if not ret:
            print("Error: Could not read frame")
            return
        
        print("Starting motion detection...")
        
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
                frame_count = 0
                start_time = time.time()
            
            # Apply preprocessing
            processed_frame = preprocess_frame(
                frame,
                target_size=PREPROCESSING["target_size"],
                normalize=PREPROCESSING["normalize"],
                denoise=PREPROCESSING["denoise"],
                enhance_contrast=PREPROCESSING["enhance_contrast"]
            )
            
            # Detect motion
            has_motion, motion_regions = motion_detector.detect(processed_frame)
            
            # Draw motion visualization
            frame = motion_detector.draw_motion(frame, motion_regions)
            
            # Update UI with tracking information
            tracking_info = {
                'tracking': True,
                'fps': fps,
                'motion_detected': has_motion
            }
            
            key = ui.update_display(frame, tracking_info)
            
            # Handle key presses
            if key == ord('q'):
                break
    
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Stack trace:")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        if 'cap' in locals():
            cap.release()
        if 'ui' in locals():
            ui.cleanup()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 