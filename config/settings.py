# Video settings
# Turn resolution down when there are less resources...
VIDEO_SOURCE = 0  # Use 0 for webcam, or provide path to video file
FRAME_WIDTH = 640 # 640 1280
FRAME_HEIGHT = 480 # 480 720
FPS = 30 # 30 this setting in not being used

# Tracking settings
TRACKER_TYPE = "TLD"  # Options: CSRT, KCF, MOSSE, MIL, BOOSTING, MEDIANFLOW, TLD
CONFIDENCE_THRESHOLD = 0.5

# Display settings
SHOW_FPS = True
SHOW_BOUNDING_BOX = True
BOUNDING_BOX_COLOR = (0, 255, 0)  # Green in BGR
BOUNDING_BOX_THICKNESS = 2

# Preprocessing settings
PREPROCESSING = {
    "enabled": True,
    "target_size": (640, 480),  # Match the video resolution
    "normalize": False,  # Disabled to prevent darkening
    "denoise": False,  # Disabled for performance
    "enhance_contrast": True,
    "gaussian_blur": {
        "enabled": False,
        "kernel_size": (5, 5),
        "sigma": 0.0
    },
    "edge_enhancement": {
        "enabled": False,
        "low_threshold": 50,
        "high_threshold": 150
    },
    "brightness_contrast": {
        "enabled": True,  # Enable brightness/contrast adjustment
        "brightness": 1.2,  # Increase brightness by 20%
        "contrast": 1.1  # Increase contrast by 10%
    }
} 