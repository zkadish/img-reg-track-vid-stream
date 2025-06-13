# Video settings
# Turn resolution down when there are less resources...
VIDEO_SOURCE = 0  # Use 0 for webcam, or provide path to video file
FRAME_WIDTH = 640 # 640 1280
FRAME_HEIGHT = 480 # 480 720
FPS = 60 # 30 this setting in not being used

# Tracking settings
TRACKER_TYPE = "CSRT"  # Options: CSRT, KCF, MOSSE
CONFIDENCE_THRESHOLD = 0.5

# Display settings
SHOW_FPS = True
SHOW_BOUNDING_BOX = True
BOUNDING_BOX_COLOR = (0, 255, 0)  # Green in BGR
BOUNDING_BOX_THICKNESS = 2

# Preprocessing settings
PREPROCESSING = {
    "enabled": True,
    "target_size": (640, 480),  # 640x640 YOLO's preferred input size
    "normalize": True,
    "denoise": False,  # Disabled by default for performance
    "enhance_contrast": True,
    "gaussian_blur": {
        "enabled": False,  # Disabled by default for performance
        "kernel_size": (5, 5),
        "sigma": 0.0
    },
    "edge_enhancement": {
        "enabled": False,  # Disabled by default for performance
        "low_threshold": 50,
        "high_threshold": 150
    },
    "brightness_contrast": {
        "enabled": False,  # Disabled by default for performance
        "brightness": 1.0,
        "contrast": 1.0
    }
} 