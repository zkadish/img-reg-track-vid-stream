import cv2
import numpy as np
from config.settings import TRACKER_TYPE

class ImageTracker:
    def __init__(self, tracker_type: str = TRACKER_TYPE):
        """
        Initialize tracking algorithm
        Args:
            tracker_type: Type of tracker to use ('CSRT', 'KCF', or 'MOSSE')
        """
        self.tracker_type = tracker_type
        self.tracker = self._create_tracker()
        self.bbox = None
        self.initialized = False

    def _create_tracker(self):
        """Create the appropriate tracker based on type"""
        if self.tracker_type == "CSRT":
            return cv2.legacy.TrackerCSRT_create()
        elif self.tracker_type == "KCF":
            return cv2.legacy.TrackerKCF_create()
        elif self.tracker_type == "MOSSE":
            return cv2.legacy.TrackerMOSSE_create()
        else:
            print(f"Warning: Unknown tracker type '{self.tracker_type}'. Using CSRT instead.")
            return cv2.legacy.TrackerCSRT_create()

    def initialize(self, frame, bbox):
        """Initialize the tracker with a frame and bounding box"""
        self.initialized = self.tracker.init(frame, bbox)
        self.bbox = bbox
        return self.initialized

    def update(self, frame):
        """Update the tracker with a new frame"""
        if not self.initialized:
            return False, None

        success, bbox = self.tracker.update(frame)
        self.bbox = bbox if success else None
        return success, bbox

    def draw_bbox(self, frame):
        """Draw the bounding box on the frame"""
        if self.bbox is not None:
            x, y, w, h = [int(v) for v in self.bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Add tracker type label
            cv2.putText(frame, f"Tracker: {self.tracker_type}", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame 