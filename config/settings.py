# Video settings
# Turn resolution down when there are less resources...
VIDEO_SOURCE = 0  # Use 0 for webcam, or provide path to video file
FRAME_WIDTH = 1280 # 640 1280
FRAME_HEIGHT = 720 # 480 720
FPS = 30 # 30 this setting in not being used

# Tracking settings
TRACKER_TYPE = "CSRT"  # Options: CSRT, KCF, MOSSE, MIL, BOOSTING, MEDIANFLOW, TLD
CONFIDENCE_THRESHOLD = 0.5

# Display settings
SHOW_FPS = True
SHOW_BOUNDING_BOX = True
BOUNDING_BOX_COLOR = (0, 255, 0)  # Green in BGR
BOUNDING_BOX_THICKNESS = 1 # this doesn't look like it's being used

# YOLO settings
YOLO_MODEL_PATH = "models/yolov8n.pt"  # Path to YOLO model weights
YOLO_CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for detections

# Motion detection settings
MOTION_DETECTION = {
    "enabled": False,  # Enable/disable motion detection
    "min_area": 100,  # Minimum area for motion detection
    "history": 5,     # Number of frames to keep in history
    "threshold": 25,  # Threshold for motion detection
    "visualization": {
        "enabled": True,  # Show motion visualization
        "color": (0, 0, 255),  # Red color for motion outlines
        "thickness": 2  # Line thickness for motion outlines
    }
} 

# Preprocessing settings
# Turn off preprocessing for performance. Turning off preprocessing also improves class detection.
PREPROCESSING = {
    "enabled": False,
    "target_size": (FRAME_WIDTH, FRAME_HEIGHT),  # Match the video resolution
    "normalize": False,  # Disabled to prevent darkening
    "denoise": False,  # Disabled for performance
    "enhance_contrast": False,
    "gaussian_blur": {
        "enabled": False, # Disabled for performance
        "kernel_size": (5, 5),
        "sigma": 0.0
    },
    "edge_enhancement": {
        "enabled": False, # Disabled for performance
        "low_threshold": 50,
        "high_threshold": 150
    },
    "brightness_contrast": {
        "enabled": False,  # Enable brightness/contrast adjustment
        "brightness": 1.2,  # Increase brightness by 20%
        "contrast": 1.1  # Increase contrast by 10%
    }
}

# Tracker settings
TRACKING = {
    "enabled": True,  # Enable/disable image tracking
    "auto_track": True,  # Automatically start tracking when object detected
    "confidence_threshold": 0.5,  # Minimum confidence to start tracking
    "visualization": {
        "enabled": True,  # Show tracking visualization
        "box_color": (255, 0, 0),  # Blue color for tracking box
        "text_color": (255, 0, 0),  # Blue color for text
        "box_thickness": 2,  # Line thickness for tracking box
        "text_scale": 0.9,  # Text size
        "text_thickness": 2  # Text thickness
    }
}