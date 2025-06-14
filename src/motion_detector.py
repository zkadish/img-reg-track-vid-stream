import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional
import time

@dataclass
class MotionRegion:
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    area: float
    contour: np.ndarray  # Store the actual contour
    velocity: Optional[Tuple[float, float]] = None
    history: List[Tuple[int, int]] = None
    center: Tuple[int, int] = None
    
    def __post_init__(self):
        if self.history is None:
            self.history = []
        x, y, w, h = self.bbox
        self.center = (x + w//2, y + h//2)
        self.history.append(self.center)

class MotionDetector:
    def __init__(self, min_area: int = 100, history: int = 5, var_threshold: float = 16.0, detect_shadows: bool = False):
        self.min_area = min_area
        self.history = history
        self.var_threshold = var_threshold
        self.detect_shadows = detect_shadows
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows
        )
        self.motion_regions = []
        self.max_history = 10  # Maximum number of previous positions to track
        
        # Visualization settings
        self.colors = {
            'contour': (0, 255, 0),    # Green
            'trail': (0, 255, 255)     # Yellow
        }
        
        # Store previous frame for motion visualization
        self.prev_frame = None
        self.motion_history = []
        self.max_motion_history = 5
    
    def detect(self, frame: np.ndarray) -> Tuple[bool, List[MotionRegion]]:
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Apply morphological operations to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Process contours and create motion regions
        motion_regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_area:
                x, y, w, h = cv2.boundingRect(contour)
                region = MotionRegion(bbox=(x, y, w, h), area=area, contour=contour)
                
                # Calculate velocity if we have previous positions
                if self.motion_regions:
                    for prev_region in self.motion_regions:
                        if self._is_same_region(region, prev_region):
                            dx = region.center[0] - prev_region.center[0]
                            dy = region.center[1] - prev_region.center[1]
                            region.velocity = (dx, dy)
                            region.history = prev_region.history[-self.max_history:] + [region.center]
                            break
                
                motion_regions.append(region)
        
        # Update motion regions history
        self.motion_regions = motion_regions
        
        # Update motion history for visualization
        if self.prev_frame is not None:
            # Calculate frame difference
            diff = cv2.absdiff(frame, self.prev_frame)
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            _, motion_mask = cv2.threshold(gray_diff, 25, 255, cv2.THRESH_BINARY)
            
            # Add to motion history
            self.motion_history.append(motion_mask)
            if len(self.motion_history) > self.max_motion_history:
                self.motion_history.pop(0)
        
        self.prev_frame = frame.copy()
        
        return len(motion_regions) > 0, motion_regions
    
    def _is_same_region(self, region1: MotionRegion, region2: MotionRegion, threshold: int = 50) -> bool:
        """Check if two regions are likely the same object based on center distance"""
        if not region1.center or not region2.center:
            return False
        dx = region1.center[0] - region2.center[0]
        dy = region1.center[1] - region2.center[1]
        distance = np.sqrt(dx*dx + dy*dy)
        return distance < threshold
    
    def draw_motion(self, frame: np.ndarray, motion_regions: List[MotionRegion]) -> np.ndarray:
        """Draw motion regions and their trajectories on the frame with modern visualization"""
        # Create a copy of the frame for drawing
        vis_frame = frame.copy()
        
        for region in motion_regions:
            # Draw actual contour of motion
            cv2.drawContours(vis_frame, [region.contour], -1, self.colors['contour'], 2)
            
            # Draw motion trail with gradient effect
            if len(region.history) > 1:
                for i in range(1, len(region.history)):
                    # Calculate alpha based on position in history
                    alpha = i / len(region.history)
                    color = (
                        int(self.colors['trail'][0] * alpha),
                        int(self.colors['trail'][1] * alpha),
                        int(self.colors['trail'][2] * alpha)
                    )
                    cv2.line(vis_frame, region.history[i-1], region.history[i], color, 2)
        
        # Blend the visualization with the original frame
        alpha = 0.7
        cv2.addWeighted(vis_frame, alpha, frame, 1-alpha, 0, frame)
        
        return frame 