import cv2
import numpy as np
from typing import Tuple, Optional
from config.settings import FRAME_WIDTH, FRAME_HEIGHT

def get_video_source(source: int = 0) -> cv2.VideoCapture:
    """Initialize and return a video capture object"""
    # First create the capture object
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise ValueError(f"Could not open video source {source}")
    
    # Get native resolution
    native_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    native_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    native_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print("\nWebcam Native Properties:")
    print(f"Resolution: {native_width}x{native_height}")
    print(f"FPS: {native_fps}")
    
    # Read a test frame to ensure camera is working
    ret, _ = cap.read()
    if not ret:
        raise ValueError("Could not read from video source")
    
    # Now try to set the resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    
    # Verify the resolution was set correctly
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print("\nResolution Settings:")
    print(f"Requested resolution: {FRAME_WIDTH}x{FRAME_HEIGHT}")
    print(f"Actual resolution: {actual_width}x{actual_height}")
    
    return cap

def read_frame(cap: cv2.VideoCapture) -> Tuple[bool, np.ndarray]:
    """Read a frame from the video capture"""
    ret, frame = cap.read()
    if not ret:
        return False, np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
    return True, frame

def resize_frame(frame: np.ndarray, width: int = None, height: int = None) -> np.ndarray:
    """Resize frame while maintaining aspect ratio"""
    if width is None and height is None:
        return frame

    h, w = frame.shape[:2]
    if width is None:
        aspect = height / float(h)
        dim = (int(w * aspect), height)
    else:
        aspect = width / float(w)
        dim = (width, int(h * aspect))

    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA) 