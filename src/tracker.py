import cv2
import numpy as np

class ImageTracker:
    def __init__(self):
        # Initialize tracking algorithm (CSRT is a good balance of speed and accuracy)
        self.tracker = cv2.TrackerCSRT_create()
        self.bbox = None
        self.initialized = False

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
        return frame 