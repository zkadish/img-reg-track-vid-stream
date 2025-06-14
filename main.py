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
        print("Initializing YOLO detector...")
        detector = ObjectDetector(confidence=0.5)
        print("Initializing tracker...")
        tracker = ImageTracker(tracker_type=TRACKER_TYPE)
        print("Initializing motion detector...")
        motion_detector = MotionDetector(min_area=100, history=5)
        
        # Initialize UI
        ui = TrackingUI()
        
        print(f"Using {TRACKER_TYPE} tracker")
        print("Press 'q' to quit")
        print("Press 'r' to reset tracking")
        
        # FPS calculation variables
        frame_count = 0
        start_time = time.time()
        fps = 0
        
        # Tracking state
        tracking = False
        current_class = None
        current_conf = None
        
        # Wait for first frame
        ret, frame = read_frame(cap)
        if not ret:
            print("Error: Could not read frame")
            return
            
        # Let user select target
        print("Select target object to track...")
        selected_roi = ui.select_target(frame)
        if selected_roi is None:
            print("Target selection cancelled")
            return
            
        # Initialize tracker with selected ROI
        tracking = tracker.initialize(frame, selected_roi)
        if not tracking:
            print("Failed to initialize tracker")
            return
            
        print("Tracking initialized. Press 'q' to quit or 'r' to reset")
        
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
            
            # Step 1: Detect motion
            has_motion, motion_regions = motion_detector.detect(processed_frame)
            
            if not tracking:
                if has_motion:
                    # Step 2: Only process regions with motion
                    for region in motion_regions:
                        x, y, w, h = region.bbox
                        roi = processed_frame[y:y+h, x:x+w]
                        if roi.size > 0:
                            # Step 3: Detect objects in motion regions
                            detections = detector.detect(roi)
                            
                            # Adjust detection coordinates to original frame
                            adjusted_detections = []
                            for bbox, conf, class_id in detections:
                                bx, by, bw, bh = bbox
                                adjusted_detections.append((
                                    (bx + x, by + y, bw, bh),
                                    conf,
                                    class_id
                                ))
                            
                            # Draw detections on original frame
                            frame = detector.draw_detections(frame, adjusted_detections)
                            
                            # If we have detections, start tracking the first one
                            if adjusted_detections:
                                bbox, confidence, class_id = adjusted_detections[0]
                                current_class = detector.class_names[class_id]
                                current_conf = confidence
                                tracking = tracker.initialize(frame, bbox, current_class, current_conf)
                                print(f"Started tracking {current_class} with {TRACKER_TYPE} tracker")
                                break
            else:
                # Update tracker
                success, bbox = tracker.update(frame)
                if success:
                    frame = tracker.draw_bbox(frame)
                else:
                    print("Lost tracking, switching back to detection")
                    tracking = False
                    current_class = None
                    current_conf = None
            
            # Draw motion visualization
            frame = motion_detector.draw_motion(frame, motion_regions)
            
            # Update UI with tracking information
            tracking_info = {
                'tracking': tracking,
                'object_class': current_class,
                'confidence': current_conf,
                'tracker_type': TRACKER_TYPE,
                'fps': fps,
                'motion_detected': has_motion
            }
            
            key = ui.update_display(frame, tracking_info)
            
            # Handle key presses
            if key == ord('q'):
                break
            elif key == ord('r'):
                tracking = False
                current_class = None
                current_conf = None
                print("Reset tracking")
                # Let user select new target
                selected_roi = ui.select_target(frame)
                if selected_roi is not None:
                    tracking = tracker.initialize(frame, selected_roi)
                    if tracking:
                        print("Tracking reinitialized")
    
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