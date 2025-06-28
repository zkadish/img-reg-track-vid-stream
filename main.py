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
        print("Press 'd' to enable debug mode")
        print("Press 'p' to toggle performance mode")
        print("Press 'c' to toggle confidence monitoring")
        print("Press 's' to toggle simultaneous tracking and recognition")
        
        # Debug mode
        debug_mode = False
        
        # Performance mode
        performance_mode = False
        
        # Performance timing
        tracking_times = []
        
        # Confidence monitoring
        confidence_check_counter = 0
        
        # Motion detection timing
        motion_start_time = None
        motion_detected_duration = 0
        motion_required_duration = 0.1  # 0.1 second of motion before enabling recognition
        
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
        
        # Test detection to verify YOLO is working
        if detector:
            print("Testing YOLO detection...")
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            test_detections = detector.detect(rgb_frame)
            print(f"Initial detection test found {len(test_detections)} objects")
            if len(test_detections) > 0:
                for bbox, conf, class_id in test_detections[:3]:  # Show first 3
                    obj_name = detector.class_names[class_id]
                    print(f"  - {obj_name}: {conf:.2f}")
        
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
            
            # Apply preprocessing only during detection phase, not tracking
            if not is_tracking:
                processed_frame = preprocess_frame(
                    frame,
                    target_size=PREPROCESSING["target_size"],
                    normalize=PREPROCESSING["normalize"],
                    denoise=PREPROCESSING["denoise"],
                    enhance_contrast=PREPROCESSING["enhance_contrast"]
                )
            else:
                # During tracking, use original frame for better performance
                processed_frame = frame
            
            # Detect motion if enabled (only during detection phase)
            has_motion = False
            if MOTION_DETECTION["enabled"] and motion_detector and not is_tracking:
                has_motion, motion_regions = motion_detector.detect(processed_frame)
                
                # Track motion duration
                current_time = time.time()
                if has_motion:
                    if motion_start_time is None:
                        motion_start_time = current_time
                        motion_detected_duration = 0
                        print("Motion detected - starting timer")
                    else:
                        motion_detected_duration = current_time - motion_start_time
                        if debug_mode:
                            print(f"Motion duration: {motion_detected_duration:.1f}s")
                else:
                    # Reset motion timer if no motion detected
                    if motion_start_time is not None:
                        print("Motion stopped - resetting timer")
                    motion_start_time = None
                    motion_detected_duration = 0
                
                # Draw motion visualization if enabled
                if MOTION_DETECTION["visualization"]["enabled"]:
                    frame = motion_detector.draw_motion(
                        frame, 
                        motion_regions,
                        color=MOTION_DETECTION["visualization"]["color"],
                        thickness=MOTION_DETECTION["visualization"]["thickness"]
                    )
            
            # Image recognition and tracking (only after motion detected for required duration)
            motion_requirement_met = (motion_detected_duration >= motion_required_duration) or is_tracking
            if (IMAGE_RECOGNITION["enabled"] and detector and motion_requirement_met and 
                (not is_tracking or IMAGE_RECOGNITION["continue_during_tracking"])):
                if TRACKING["enabled"] and tracker:
                    # Convert frame to RGB for YOLO detection
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Perform object detection on the RGB frame
                    detections = detector.detect(rgb_frame)
                    
                    # Debug output
                    if debug_mode and len(detections) > 0:
                        print(f"Found {len(detections)} detections:")
                        for i, (bbox, conf, class_id) in enumerate(detections):
                            obj_name = detector.class_names[class_id]
                            print(f"  {i+1}. {obj_name}: {conf:.2f}")
                    
                    # Check for high confidence detections if auto-track is enabled and not already tracking
                    if TRACKING["auto_track"] and TRACKING["stability"]["enabled"] and not is_tracking:
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
                                print(f"New detection: {current_object} with confidence {conf:.2f}")
                            else:
                                # Update consecutive detections
                                consecutive_detections += 1
                                last_detection = (bbox, conf, current_object)
                                print(f"Consecutive detection #{consecutive_detections}: {current_object} ({conf:.2f})")
                            
                            # Check if stability conditions are met
                            time_elapsed = current_time - detection_start_time
                            print(f"Stability check: {time_elapsed:.1f}s elapsed, {consecutive_detections} detections")
                            
                            if (time_elapsed >= TRACKING["stability"]["delay_seconds"] and 
                                consecutive_detections >= TRACKING["stability"]["min_detections"]):
                                # Start tracking - use BGR frame for tracker initialization
                                is_tracking = True
                                tracked_bbox = bbox
                                tracked_object = current_object
                                tracked_confidence = conf
                                print(f"Detected {tracked_object} with confidence {conf:.2f}")
                                print(f"Stable detection for {time_elapsed:.1f} seconds")
                                print("Starting tracking...")
                                
                                # Initialize tracker with BGR frame (not RGB)
                                success = tracker.initialize(frame, tracked_bbox)
                                if success:
                                    print("Tracker initialized successfully")
                                    # Reset confidence check counter
                                    confidence_check_counter = 0
                                    # Stop motion detection but keep recognition if configured
                                    MOTION_DETECTION["enabled"] = False
                                    if not IMAGE_RECOGNITION["continue_during_tracking"]:
                                        IMAGE_RECOGNITION["enabled"] = False
                                        print("Image recognition and motion detection stopped, continuing to track object")
                                    else:
                                        print("Motion detection stopped, continuing with tracking and recognition")
                                else:
                                    print("Failed to initialize tracker")
                                    is_tracking = False
                                    tracked_bbox = None
                                    tracked_object = None
                                    tracked_confidence = 0
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
                else:
                    # Just perform detection without tracking
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    detections = detector.detect(rgb_frame)
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
                # Measure tracking performance
                if performance_mode:
                    track_start = time.time()
                
                # Update tracker
                success, bbox = tracker.update(frame)
                
                if performance_mode:
                    track_time = (time.time() - track_start) * 1000  # Convert to ms
                    tracking_times.append(track_time)
                    if len(tracking_times) > 30:  # Keep last 30 measurements
                        tracking_times.pop(0)
                    avg_track_time = sum(tracking_times) / len(tracking_times)
                    print(f"Tracking time: {track_time:.1f}ms (avg: {avg_track_time:.1f}ms)")
                
                if success:
                    tracked_bbox = bbox
                    
                    # Check confidence periodically if enabled
                    if (TRACKING["confidence_monitoring"]["enabled"] and 
                        detector and 
                        TRACKING["confidence_monitoring"]["stop_on_low_confidence"]):
                        
                        confidence_check_counter += 1
                        if confidence_check_counter >= TRACKING["confidence_monitoring"]["check_interval"]:
                            confidence_check_counter = 0
                            
                            # Extract the tracked region and run detection on it
                            x, y, w, h = [int(v) for v in bbox]
                            
                            # Ensure bbox is within frame bounds
                            x = max(0, min(x, frame.shape[1] - 1))
                            y = max(0, min(y, frame.shape[0] - 1))
                            w = max(1, min(w, frame.shape[1] - x))
                            h = max(1, min(h, frame.shape[0] - y))
                            
                            if w > 10 and h > 10:  # Only check if region is large enough
                                # Extract tracked region
                                tracked_region = frame[y:y+h, x:x+w]
                                
                                # Convert to RGB for YOLO
                                rgb_region = cv2.cvtColor(tracked_region, cv2.COLOR_BGR2RGB)
                                
                                # Run detection on the tracked region
                                region_detections = detector.detect(rgb_region)
                                
                                # Check if our tracked object is still detected with sufficient confidence
                                object_still_detected = False
                                current_confidence = 0
                                
                                for det_bbox, conf, class_id in region_detections:
                                    detected_object = detector.class_names[class_id]
                                    if (detected_object == tracked_object and 
                                        conf >= TRACKING["confidence_monitoring"]["min_confidence"]):
                                        object_still_detected = True
                                        current_confidence = conf
                                        # Update tracked confidence
                                        tracked_confidence = conf
                                        if debug_mode:
                                            print(f"Confidence check: {tracked_object} still detected with {conf:.2f}")
                                        break
                                
                                if not object_still_detected:
                                    print(f"Confidence dropped below {TRACKING['confidence_monitoring']['min_confidence']:.2f} - stopping tracking")
                                    
                                    # Check if auto-reset is enabled
                                    if TRACKING["auto_reset_on_failure"]:
                                        # Auto-reset the entire application
                                        reset_state = reset_application_state()
                                        is_tracking = reset_state["is_tracking"]
                                        tracked_bbox = reset_state["tracked_bbox"]
                                        tracked_object = reset_state["tracked_object"]
                                        tracked_confidence = reset_state["tracked_confidence"]
                                        detection_start_time = reset_state["detection_start_time"]
                                        last_detection = reset_state["last_detection"]
                                        consecutive_detections = reset_state["consecutive_detections"]
                                        confidence_check_counter = reset_state["confidence_check_counter"]
                                        has_motion = reset_state["has_motion"]
                                        motion_bbox = reset_state["motion_bbox"]
                                        motion_start_time = reset_state["motion_start_time"]
                                        motion_detected_duration = reset_state["motion_detected_duration"]
                                        frame_count = reset_state["frame_count"]
                                        start_time = reset_state["start_time"]
                                        fps = reset_state["fps"]
                                        
                                        # Reinitialize components
                                        print("Reinitializing components after confidence failure...")
                                        
                                        # Reset motion detector
                                        if motion_detector is not None:
                                            motion_detector = MotionDetector(
                                                min_area=MOTION_DETECTION["min_area"],
                                                var_threshold=MOTION_DETECTION["threshold"]
                                            )
                                        
                                        # Reset tracker
                                        if tracker is not None:
                                            tracker = ImageTracker()
                                        
                                        # Re-enable recognition and motion detection
                                        IMAGE_RECOGNITION["enabled"] = True
                                        MOTION_DETECTION["enabled"] = True
                                        
                                        print("Application reset complete - ready for new detection")
                                    else:
                                        # Original behavior - just reset tracking state
                                        is_tracking = False
                                        tracked_bbox = None
                                        tracked_object = None
                                        tracked_confidence = 0
                                        # Re-enable recognition and motion detection if needed
                                        if not IMAGE_RECOGNITION["continue_during_tracking"]:
                                            IMAGE_RECOGNITION["enabled"] = True
                                        MOTION_DETECTION["enabled"] = True
                                        print("Low confidence - re-enabling detection systems")
                                    continue  # Skip drawing this frame
                    
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
                    
                    # Check if auto-reset is enabled
                    if TRACKING["auto_reset_on_failure"]:
                        # Auto-reset the entire application
                        reset_state = reset_application_state()
                        is_tracking = reset_state["is_tracking"]
                        tracked_bbox = reset_state["tracked_bbox"]
                        tracked_object = reset_state["tracked_object"]
                        tracked_confidence = reset_state["tracked_confidence"]
                        detection_start_time = reset_state["detection_start_time"]
                        last_detection = reset_state["last_detection"]
                        consecutive_detections = reset_state["consecutive_detections"]
                        confidence_check_counter = reset_state["confidence_check_counter"]
                        has_motion = reset_state["has_motion"]
                        motion_bbox = reset_state["motion_bbox"]
                        motion_start_time = reset_state["motion_start_time"]
                        motion_detected_duration = reset_state["motion_detected_duration"]
                        frame_count = reset_state["frame_count"]
                        start_time = reset_state["start_time"]
                        fps = reset_state["fps"]
                        
                        # Reinitialize components
                        print("Reinitializing components after tracking failure...")
                        
                        # Reset motion detector
                        if motion_detector is not None:
                            motion_detector = MotionDetector(
                                min_area=MOTION_DETECTION["min_area"],
                                var_threshold=MOTION_DETECTION["threshold"]
                            )
                        
                        # Reset tracker
                        if tracker is not None:
                            tracker = ImageTracker()
                        
                        # Re-enable recognition and motion detection
                        IMAGE_RECOGNITION["enabled"] = True
                        MOTION_DETECTION["enabled"] = True
                        
                        print("Application reset complete - ready for new detection")
                    else:
                        # Original behavior - just reset tracking state
                        is_tracking = False
                        tracked_bbox = None
                        tracked_object = None
                        tracked_confidence = 0
                        # Re-enable recognition and motion detection if needed
                        if not IMAGE_RECOGNITION["continue_during_tracking"]:
                            IMAGE_RECOGNITION["enabled"] = True
                        MOTION_DETECTION["enabled"] = True
                        print("Tracking lost - re-enabling detection systems")
            
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
                "motion_enabled": MOTION_DETECTION["enabled"],
                "confidence_monitoring_enabled": TRACKING["confidence_monitoring"]["enabled"],
                "simultaneous_mode": IMAGE_RECOGNITION["continue_during_tracking"],
                "auto_reset_enabled": TRACKING["auto_reset_on_failure"],
                "debug_mode": debug_mode,
                "performance_mode": performance_mode,
                "motion_detected_duration": motion_detected_duration,
                "motion_required_duration": motion_required_duration,
                "tracker_type": TRACKER_TYPE,
                "tracking": is_tracking,
                "object_class": tracked_object,
                "confidence": tracked_confidence,
                "fps": fps
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
                # Reset confidence monitoring
                confidence_check_counter = 0
                # Reset motion detection state
                has_motion = False
                motion_bbox = None
                # Reset motion timing
                motion_start_time = None
                motion_detected_duration = 0
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
                # Enable recognition and motion detection
                IMAGE_RECOGNITION["enabled"] = True
                MOTION_DETECTION["enabled"] = True
                print("Recognition and motion detection enabled")
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
                    
                    # Re-enable recognition and motion detection
                    IMAGE_RECOGNITION["enabled"] = True
                    MOTION_DETECTION["enabled"] = True
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
                    # Also enable motion detection when recognition is enabled
                    MOTION_DETECTION["enabled"] = True
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
                    # Reset motion timing
                    motion_start_time = None
                    motion_detected_duration = 0
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
                    # Reset motion timing
                    motion_start_time = None
                    motion_detected_duration = 0
                else:
                    print("Motion detection disabled")
                    # Reset motion detection state
                    has_motion = False
                    motion_bbox = None
                    # Reset motion timing
                    motion_start_time = None
                    motion_detected_duration = 0
            elif key == ord('d'):
                # Toggle debug mode
                debug_mode = not debug_mode
                print(f"Debug mode: {'ON' if debug_mode else 'OFF'}")
            elif key == ord('p'):
                # Toggle performance mode
                performance_mode = not performance_mode
                print(f"Performance mode: {'ON' if performance_mode else 'OFF'}")
            elif key == ord('c'):
                # Toggle confidence monitoring
                TRACKING["confidence_monitoring"]["enabled"] = not TRACKING["confidence_monitoring"]["enabled"]
                print(f"Confidence monitoring: {'ON' if TRACKING['confidence_monitoring']['enabled'] else 'OFF'}")
            elif key == ord('s'):
                # Toggle simultaneous tracking and recognition mode
                IMAGE_RECOGNITION["continue_during_tracking"] = not IMAGE_RECOGNITION["continue_during_tracking"]
                if IMAGE_RECOGNITION["continue_during_tracking"]:
                    print("Simultaneous mode enabled - tracking and recognition will run together")
                    # Enable recognition if it was disabled
                    IMAGE_RECOGNITION["enabled"] = True
                else:
                    print("Simultaneous mode disabled - recognition will stop during tracking")
                    # If currently tracking, disable recognition
                    if is_tracking:
                        IMAGE_RECOGNITION["enabled"] = False
            elif key == ord('a'):
                # Toggle auto-reset on tracking failure
                TRACKING["auto_reset_on_failure"] = not TRACKING["auto_reset_on_failure"]
                print(f"Auto-reset on tracking failure: {'ON' if TRACKING['auto_reset_on_failure'] else 'OFF'}")
                if TRACKING["auto_reset_on_failure"]:
                    print("  App will reset completely when tracking fails")
                else:
                    print("  App will only re-enable detection systems when tracking fails")
    
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

def reset_application_state():
    """Reset all application state variables and components"""
    print("Auto-resetting application...")
    
    # This function returns a dictionary of reset values
    # that will be used to update the main loop variables
    return {
        "is_tracking": False,
        "tracked_bbox": None,
        "tracked_object": None,
        "tracked_confidence": 0,
        "detection_start_time": None,
        "last_detection": None,
        "consecutive_detections": 0,
        "confidence_check_counter": 0,
        "has_motion": False,
        "motion_bbox": None,
        "motion_start_time": None,
        "motion_detected_duration": 0,
        "frame_count": 0,
        "start_time": time.time(),
        "fps": 0,
        "reset_components": True  # Flag to reinitialize components
    }

if __name__ == "__main__":
    main() 