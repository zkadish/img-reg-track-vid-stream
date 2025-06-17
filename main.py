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
        print("Press 'i' to toggle recognition on/off")
        print("Press 'm' to toggle motion detection on/off")
        
        # FPS calculation variables
        frame_count = 0
        start_time = time.time()
        fps = 0
        
        # Tracking state
        is_tracking = False
        tracked_bbox = None
        tracked_object = None
        tracked_confidence = 0
        
        # Detection stability state
        detection_start_time = None
        last_detection = None
        consecutive_detections = 0
        
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
            
            # Process frame
            processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Image recognition and tracking
            if IMAGE_RECOGNITION["enabled"] and detector and not is_tracking:
                if TRACKING["enabled"] and tracker:
                    # Perform object detection on the entire frame
                    detections = detector.detect(processed_frame)
                    
                    # Check for high confidence detections if auto-track is enabled
                    if TRACKING["auto_track"] and TRACKING["stability"]["enabled"]:
                        current_time = time.time()
                        
                        # Find best detection
                        best_detection = None
                        for bbox, conf, class_id in detections:
                            if conf >= TRACKING["confidence_threshold"]:
                                best_detection = (bbox, conf, class_id)
                                break
                        
                        if best_detection:
                            bbox, conf, class_id = best_detection
                            current_object = detector.class_names[class_id]
                            
                            # Check if this is a new detection or continuation
                            if (last_detection is None or 
                                (TRACKING["stability"]["same_class"] and 
                                 current_object != last_detection[2])):
                                # Reset stability check for new detection
                                detection_start_time = current_time
                                consecutive_detections = 1
                                last_detection = (bbox, conf, current_object)
                            else:
                                # Update consecutive detections
                                consecutive_detections += 1
                                last_detection = (bbox, conf, current_object)
                            
                            # Check if stability conditions are met
                            time_elapsed = current_time - detection_start_time
                            if (time_elapsed >= TRACKING["stability"]["delay_seconds"] and 
                                consecutive_detections >= TRACKING["stability"]["min_detections"]):
                                # Start tracking
                                is_tracking = True
                                tracked_bbox = bbox
                                tracked_object = current_object
                                tracked_confidence = conf
                                print(f"Detected {tracked_object} with confidence {conf:.2f}")
                                print(f"Stable detection for {time_elapsed:.1f} seconds")
                                print("Starting tracking...")
                                tracker.initialize(frame, tracked_bbox)
                                # Stop image recognition
                                IMAGE_RECOGNITION["enabled"] = False
                                print("Image recognition stopped, continuing to track object")
                        else:
                            # Reset stability check if no good detection
                            detection_start_time = None
                            last_detection = None
                            consecutive_detections = 0
                    
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
                        
                        # Draw stability status if checking stability
                        if TRACKING["stability"]["enabled"] and detection_start_time is not None:
                            time_elapsed = time.time() - detection_start_time
                            status = f"Stability: {time_elapsed:.1f}s, {consecutive_detections} detections"
                            cv2.putText(
                                frame,
                                status,
                                (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 255, 255),
                                2
                            )
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
            
            # Update tracking if active
            if is_tracking and tracker:
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
            
            # Update tracking info
            tracking_info = {
                "is_tracking": is_tracking,
                "tracked_bbox": tracked_bbox,
                "tracked_object": tracked_object,
                "tracked_confidence": tracked_confidence,
                "detection_start_time": detection_start_time,
                "last_detection": last_detection,
                "consecutive_detections": consecutive_detections,
                "tracking_enabled": TRACKING["enabled"],
                "recognition_enabled": IMAGE_RECOGNITION["enabled"],
                "motion_enabled": MOTION_DETECTION["enabled"]  # Add motion status
            }
            
            # Update display
            key = ui.update_display(frame, tracking_info)
            
            # Handle key presses
            if key == ord('q'):
                break
            elif key == ord('r'):
                print("Resetting...")
                # Reset tracking state
                is_tracking = False
                tracked_bbox = None
                tracked_object = None
                tracked_confidence = 0
                # Reset stability check
                detection_start_time = None
                last_detection = None
                consecutive_detections = 0
                # Reset motion detection state
                has_motion = False
                motion_bbox = None
                # Reset FPS calculation
                frame_count = 0
                start_time = time.time()
                fps = 0
                # Reset tracking info
                tracking_info = {
                    "tracked_object": None,
                    "tracked_confidence": 0,
                    "tracked_bbox": None,
                    "is_tracking": False,
                    "detection_start_time": None,
                    "last_detection": None,
                    "consecutive_detections": 0,
                    "tracking_enabled": TRACKING["enabled"],
                    "recognition_enabled": IMAGE_RECOGNITION["enabled"],
                    "motion_enabled": MOTION_DETECTION["enabled"]
                }
                # Reset motion detector
                if motion_detector is not None:
                    print("Reinitializing motion detector...")
                    motion_detector = MotionDetector(
                        min_area=MOTION_DETECTION["min_area"],
                        var_threshold=MOTION_DETECTION["threshold"]
                    )
                # Reset tracker
                if tracker is not None:
                    tracker = ImageTracker()
                # Enable recognition
                IMAGE_RECOGNITION["enabled"] = True
                print("Recognition enabled")
                # Initialize detector if needed
                if detector is None:
                    print("Initializing YOLO detector...")
                    detector = ObjectImageRecognition(confidence=IMAGE_RECOGNITION["confidence_threshold"])
            elif key == ord('t'):
                # Toggle tracking
                TRACKING["enabled"] = not TRACKING["enabled"]
                if TRACKING["enabled"]:
                    print("Reinitializing components...")
                    
                    # Reset tracking state
                    is_tracking = False
                    tracked_bbox = None
                    tracked_object = None
                    tracked_confidence = 0
                    
                    # Reinitialize tracker
                    print("Initializing tracker...")
                    tracker = ImageTracker(tracker_type=TRACKER_TYPE)
                    
                    print("Tracking enabled: Starting fresh detection and tracking")
                else:
                    print("Tracking disabled")
                    is_tracking = False
                    tracked_bbox = None
                    tracked_object = None
                    tracked_confidence = 0
            elif key == ord('i'):
                # Toggle recognition
                IMAGE_RECOGNITION["enabled"] = not IMAGE_RECOGNITION["enabled"]
                if IMAGE_RECOGNITION["enabled"]:
                    print("Recognition enabled")
                    # Reset tracking state
                    is_tracking = False
                    tracked_bbox = None
                    tracked_object = None
                    tracked_confidence = 0
                    # Reset stability check
                    detection_start_time = None
                    last_detection = None
                    consecutive_detections = 0
                    # Reset motion detection state
                    has_motion = False
                    motion_bbox = None
                    # Reset FPS calculation
                    frame_count = 0
                    start_time = time.time()
                    fps = 0
                    # Reset tracking info
                    tracking_info = {
                        "tracked_object": None,
                        "tracked_confidence": 0,
                        "tracked_bbox": None,
                        "is_tracking": False,
                        "detection_start_time": None,
                        "last_detection": None,
                        "consecutive_detections": 0,
                        "tracking_enabled": TRACKING["enabled"],
                        "recognition_enabled": IMAGE_RECOGNITION["enabled"]
                    }
                    # Reset motion detector
                    if motion_detector is not None:
                        print("Reinitializing motion detector...")
                        motion_detector = MotionDetector(
                            min_area=MOTION_DETECTION["min_area"],
                            var_threshold=MOTION_DETECTION["threshold"]
                        )
                    # Reset tracker
                    if tracker is not None:
                        tracker = ImageTracker()
                    # Initialize detector if needed
                    if detector is None:
                        print("Initializing YOLO detector...")
                        detector = ObjectImageRecognition(confidence=IMAGE_RECOGNITION["confidence_threshold"])
                else:
                    print("Recognition disabled")
                    # Reset tracking if recognition is disabled
                    is_tracking = False
                    tracked_bbox = None
                    tracked_object = None
                    tracked_confidence = 0
            elif key == ord('m'):
                # Toggle motion detection
                MOTION_DETECTION["enabled"] = not MOTION_DETECTION["enabled"]
                if MOTION_DETECTION["enabled"]:
                    print("Motion detection enabled")
                    # Initialize motion detector if needed
                    if motion_detector is None:
                        print("Initializing motion detector...")
                        motion_detector = MotionDetector(
                            min_area=MOTION_DETECTION["min_area"],
                            var_threshold=MOTION_DETECTION["threshold"]
                        )
                    # Reset motion detection state
                    has_motion = False
                    motion_bbox = None
                else:
                    print("Motion detection disabled")
                    # Reset motion detection state
                    has_motion = False
                    motion_bbox = None
    
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