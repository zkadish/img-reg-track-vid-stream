import cv2
import numpy as np
from config.settings import TRACKER_TYPE

class ImageTracker:
    def __init__(self, tracker_type: str = TRACKER_TYPE):
        """
        Initialize tracking algorithm
        Args:
            tracker_type: Type of tracker to use. Available options:
                - 'CSRT': Discriminative Correlation Filter with Channel and Spatial Reliability
                - 'KCF': Kernelized Correlation Filter
                - 'MOSSE': Minimum Output Sum of Squared Error
                - 'MIL': Multiple Instance Learning
                - 'BOOSTING': AdaBoost classifier
                - 'MEDIANFLOW': Median Flow tracker
                - 'TLD': Tracking, Learning and Detection
        """
        self.tracker_type = tracker_type
        self.tracker = self._create_tracker()
        self.bbox = None
        self.initialized = False
        self.object_class = None
        self.confidence = None

    def _create_tracker(self):
        """Create the appropriate tracker based on type"""
        try:
            # Try new OpenCV 4.x API first
            if self.tracker_type == "MIL":
                return cv2.TrackerMIL_create()
            elif self.tracker_type == "GOTURN":
                return cv2.TrackerGOTURN_create()
            elif self.tracker_type == "DASIAMRPN":
                return cv2.TrackerDaSiamRPN_create()
            elif self.tracker_type == "NANO":
                return cv2.TrackerNano_create()
            elif self.tracker_type == "VIT":
                return cv2.TrackerVit_create()
            else:
                # Try legacy API for older trackers
                try:
                    if self.tracker_type == "CSRT":
                        return cv2.legacy.TrackerCSRT_create()
                    elif self.tracker_type == "KCF":
                        return cv2.legacy.TrackerKCF_create()
                    elif self.tracker_type == "MOSSE":
                        return cv2.legacy.TrackerMOSSE_create()
                    elif self.tracker_type == "BOOSTING":
                        return cv2.legacy.TrackerBoosting_create()
                    elif self.tracker_type == "MEDIANFLOW":
                        return cv2.legacy.TrackerMedianFlow_create()
                    elif self.tracker_type == "TLD":
                        return cv2.legacy.TrackerTLD_create()
                except AttributeError:
                    pass
                
                # Default to MIL if others fail
                print(f"Warning: Tracker '{self.tracker_type}' not available. Using MIL instead.")
                return cv2.TrackerMIL_create()
                
        except Exception as e:
            print(f"Error creating tracker: {e}")
            print("Falling back to MIL tracker")
            return cv2.TrackerMIL_create()

    def initialize(self, frame, bbox, object_class=None, confidence=None):
        """Initialize the tracker with a frame and bounding box"""
        self.initialized = self.tracker.init(frame, bbox)
        self.bbox = bbox
        self.object_class = object_class
        self.confidence = confidence
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
            
            # Prepare label text
            label_parts = []
            if self.object_class:
                label_parts.append(f" {self.object_class}")
            if self.confidence is not None:
                label_parts.append(f"{self.confidence:.2f}")
            label_parts.append(f"{self.tracker_type} ")
            
            # Draw label with background
            label = " | ".join(label_parts)
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x, y - label_h - 10), (x + label_w, y), (0, 255, 0), -1)
            cv2.putText(frame, label, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        return frame 