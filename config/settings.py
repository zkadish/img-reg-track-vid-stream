# Video settings
VIDEO_SOURCE = 0  # 0 for default webcam
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

# Tracking settings
TRACKER_TYPE = "CSRT"  # Options: CSRT, KCF, MOSSE
CONFIDENCE_THRESHOLD = 0.5

# Display settings
SHOW_FPS = True
SHOW_BOUNDING_BOX = True
BOUNDING_BOX_COLOR = (0, 255, 0)  # Green in BGR
BOUNDING_BOX_THICKNESS = 2 