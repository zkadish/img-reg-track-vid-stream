import cv2
import numpy as np
import time
from pathlib import Path
from src.image_recognition import ObjectImageRecognition
from src.tracker import ImageTracker
from src.ui import TrackingUI
from src.motion_detector import MotionDetector
from utils.video_utils import get_video_source, read_frame
from utils.preprocessing import preprocess_frame
from config.settings import (
    VIDEO_SOURCE, PREPROCESSING, TRACKER_TYPE, 
    MOTION_DETECTION, TRACKING, IMAGE_RECOGNITION
)

def main():
    try:
        # Initialize video capture
        cap = get_video_source(VIDEO_SOURCE)
        
        # Initialize components
        detector = None
        if IMAGE_RECOGNITION["enabled"]:
            print("Initializing YOLO detector...")
            detector = ObjectImageRecognition(confidence=IMAGE_RECOGNITION["confidence_threshold"])
        
        # Initialize tracker if enabled
        tracker = None
        if TRACKING["enabled"]:
            print("Initializing tracker...")
            tracker = ImageTracker(tracker_type=TRACKER_TYPE)
        
        # Initialize motion detector if enabled
        motion_detector = None
        if MOTION_DETECTION["enabled"]:
            print("Initializing motion detector...")
            motion_detector = MotionDetector(
                min_area=MOTION_DETECTION["min_area"],
                history=MOTION_DETECTION["history"]
            )
        
        # Initialize UI
        ui = TrackingUI()
        
        print("Press 'q' to quit")
        print("Press 'r' to reset and start detection again")
        print("Press 't' to toggle tracking on/off")
        print("Press 'd' to toggle detection on/off")
        
        # FPS calculation variables
        frame_count = 0
        start_time = time.time()
        fps = 0
        
        # Tracking state
        is_tracking = False
        tracked_bbox = None
        tracked_object = None
        tracked_confidence = 0
        
        # Wait for first frame
        ret, frame = read_frame(cap)
        if not ret:
            print("Error: Could not read frame")
            return
        
        print("Starting image recognition and tracking...")
        
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
            
            # Detect motion if enabled
            has_motion = False
            if MOTION_DETECTION["enabled"] and motion_detector:
                has_motion, motion_regions = motion_detector.detect(processed_frame)
                
                # Draw motion visualization if enabled
                if MOTION_DETECTION["visualization"]["enabled"]:
                    frame = motion_detector.draw_motion(
                        frame, 
                        motion_regions,
                        color=MOTION_DETECTION["visualization"]["color"],
                        thickness=MOTION_DETECTION["visualization"]["thickness"]
                    )
            
            if IMAGE_RECOGNITION["enabled"] and detector:
                if TRACKING["enabled"] and tracker:
                    if not is_tracking:
                        # Perform object detection on the entire frame
                        detections = detector.detect(processed_frame)
                        
                        # Check for high confidence detections if auto-track is enabled
                        if TRACKING["auto_track"]:
                            for bbox, conf, class_id in detections:
                                if conf >= TRACKING["confidence_threshold"]:
                                    # Start tracking
                                    is_tracking = True
                                    tracked_bbox = bbox
                                    tracked_object = detector.class_names[class_id]
                                    tracked_confidence = conf
                                    print(f"Detected {tracked_object} with confidence {conf:.2f}")
                                    print("Starting tracking...")
                                    tracker.initialize(frame, tracked_bbox)
                                    break
                        
                        # Draw detections on frame if visualization is enabled
                        if IMAGE_RECOGNITION["visualization"]["enabled"]:
                            frame = detector.draw_detections(
                                frame, 
                                detections,
                                box_color=IMAGE_RECOGNITION["visualization"]["box_color"],
                                text_color=IMAGE_RECOGNITION["visualization"]["text_color"],
                                box_thickness=IMAGE_RECOGNITION["visualization"]["box_thickness"],
                                text_scale=IMAGE_RECOGNITION["visualization"]["text_scale"],
                                text_thickness=IMAGE_RECOGNITION["visualization"]["text_thickness"]
                            )
                    else:
                        # Update tracker
                        success, bbox = tracker.update(frame)
                        if success:
                            tracked_bbox = bbox
                            # Draw tracking box if visualization is enabled
                            if TRACKING["visualization"]["enabled"]:
                                x, y, w, h = [int(v) for v in bbox]
                                cv2.rectangle(
                                    frame, 
                                    (x, y), 
                                    (x + w, y + h), 
                                    TRACKING["visualization"]["box_color"],
                                    TRACKING["visualization"]["box_thickness"]
                                )
                                # Draw object info
                                text = f"{tracked_object}: {tracked_confidence:.2f}"
                                cv2.putText(
                                    frame, 
                                    text, 
                                    (x, y - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 
                                    TRACKING["visualization"]["text_scale"],
                                    TRACKING["visualization"]["text_color"],
                                    TRACKING["visualization"]["text_thickness"]
                                )
                        else:
                            print("Tracking lost")
                            is_tracking = False
                            tracked_bbox = None
                            tracked_object = None
                            tracked_confidence = 0
                else:
                    # Just perform detection without tracking
                    detections = detector.detect(processed_frame)
                    if IMAGE_RECOGNITION["visualization"]["enabled"]:
                        frame = detector.draw_detections(
                            frame, 
                            detections,
                            box_color=IMAGE_RECOGNITION["visualization"]["box_color"],
                            text_color=IMAGE_RECOGNITION["visualization"]["text_color"],
                            box_thickness=IMAGE_RECOGNITION["visualization"]["box_thickness"],
                            text_scale=IMAGE_RECOGNITION["visualization"]["text_scale"],
                            text_thickness=IMAGE_RECOGNITION["visualization"]["text_thickness"]
                        )
            
            # Update UI with tracking information
            tracking_info = {
                'tracking': is_tracking and TRACKING["enabled"],
                'detection': IMAGE_RECOGNITION["enabled"],
                'fps': fps,
                'motion_detected': has_motion,
                'detected_object': tracked_object,
                'detection_confidence': tracked_confidence
            }
            
            key = ui.update_display(frame, tracking_info)
            
            # Handle key presses
            if key == ord('q'):
                break
            elif key == ord('r'):
                # Reset tracking state
                is_tracking = False
                tracked_bbox = None
                tracked_object = None
                tracked_confidence = 0
                print("Reset: Starting detection again")
            elif key == ord('t'):
                # Toggle tracking
                TRACKING["enabled"] = not TRACKING["enabled"]
                if TRACKING["enabled"]:
                    print("Tracking enabled")
                    if tracker is None:
                        print("Initializing tracker...")
                        tracker = ImageTracker(tracker_type=TRACKER_TYPE)
                else:
                    print("Tracking disabled")
                    is_tracking = False
                    tracked_bbox = None
                    tracked_object = None
                    tracked_confidence = 0
            elif key == ord('d'):
                # Toggle detection
                IMAGE_RECOGNITION["enabled"] = not IMAGE_RECOGNITION["enabled"]
                if IMAGE_RECOGNITION["enabled"]:
                    print("Detection enabled")
                    if detector is None:
                        print("Initializing YOLO detector...")
                        detector = ObjectImageRecognition(confidence=IMAGE_RECOGNITION["confidence_threshold"])
                else:
                    print("Detection disabled")
                    # Reset tracking if detection is disabled
                    is_tracking = False
                    tracked_bbox = None
                    tracked_object = None
                    tracked_confidence = 0
    
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