import cv2
import numpy as np
from typing import Tuple, Optional, List
import time

class TrackingUI:
    def __init__(self, window_name: str = "Object Tracking"):
        self.window_name = window_name
        self.selection_start = None
        self.selection_end = None
        self.is_selecting = False
        self.selected_roi = None
        self.info_text = []
        self.tracking_enabled = False
        self.recognition_enabled = False
        self.motion_enabled = False
        
        # Create window and set mouse callback
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self._mouse_callback)
        
        # UI colors
        self.colors = {
            'selection': (0, 255, 0),  # Green
            'text': (255, 255, 255),   # White
            'background': (0, 0, 0),    # Black
            'highlight': (0, 165, 255)  # Orange
        }

    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for ROI selection"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.is_selecting = True
            self.selection_start = (x, y)
            self.selection_end = None
            self.selected_roi = None
        elif event == cv2.EVENT_MOUSEMOVE and self.is_selecting:
            self.selection_end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.is_selecting = False
            if self.selection_start and self.selection_end:
                # Ensure coordinates are in correct order
                x1, y1 = self.selection_start
                x2, y2 = self.selection_end
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                self.selected_roi = (x1, y1, x2 - x1, y2 - y1)

    def select_target(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Display the frame and let user select a target region
        Returns the selected ROI as (x, y, width, height) or None if cancelled
        """
        self.info_text = [
            "Select target object:",
            "- Click and drag to select region",
            "- Press 'Enter' to confirm selection",
            "- Press 'Esc' to cancel"
        ]
        
        while True:
            display_frame = frame.copy()
            
            # Draw selection rectangle if in progress
            if self.is_selecting and self.selection_start and self.selection_end:
                cv2.rectangle(display_frame, self.selection_start, self.selection_end,
                            self.colors['selection'], 2)
            
            # Draw selected ROI if exists
            if self.selected_roi:
                x, y, w, h = self.selected_roi
                cv2.rectangle(display_frame, (x, y), (x + w, y + h),
                            self.colors['highlight'], 2)
            
            # Draw info text
            self._draw_info_text(display_frame)
            
            # Show frame
            cv2.imshow(self.window_name, display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                return None
            elif key == 13 and self.selected_roi:  # Enter
                return self.selected_roi

    def _draw_info_text(self, frame: np.ndarray):
        """Draw info text on the frame"""
        if not self.info_text:
            return
            
        # Calculate text dimensions
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 1
        line_spacing = 25
        
        # Find the longest text for width calculation
        max_width = 0
        for text in self.info_text:
            (width, height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
            max_width = max(max_width, width)
        
        # Calculate border dimensions with padding
        padding = 20
        border_width = max_width + padding * 2
        border_height = len(self.info_text) * line_spacing + padding * 2
        
        # Position for info text (top right)
        y_start = 10
        x_start = frame.shape[1] - border_width - 10  # 10 pixels from right edge
        
        # Draw border
        cv2.rectangle(frame, 
                     (x_start, y_start),
                     (x_start + border_width, y_start + border_height),
                     (255, 255, 255),  # White border
                     1)  # Border thickness
        
        # Add each line of text
        for i, text in enumerate(self.info_text):
            y = y_start + padding + i * line_spacing
            cv2.putText(frame,
                       text,
                       (x_start + padding, y),
                       font,
                       font_scale,
                       (255, 255, 255),
                       font_thickness)

    def update_display(self, frame: np.ndarray, tracking_info: dict):
        """
        Update the display with tracking information
        tracking_info should contain:
        - 'tracking': bool
        - 'object_class': str
        - 'confidence': float
        - 'tracker_type': str
        - 'fps': float
        """
        display_frame = frame.copy()
        
        # Update tracking status
        if tracking_info and "tracking_enabled" in tracking_info:
            self.tracking_enabled = tracking_info["tracking_enabled"]
        
        # Update recognition status
        if tracking_info and "recognition_enabled" in tracking_info:
            self.recognition_enabled = tracking_info["recognition_enabled"]
        
        # Update motion status
        if tracking_info and "motion_enabled" in tracking_info:
            self.motion_enabled = tracking_info["motion_enabled"]
        
        # Calculate stability time
        stability_time = 0.0
        if tracking_info.get('detection_start_time') is not None:
            stability_time = time.time() - tracking_info['detection_start_time']
        
        # Update info text based on tracking status
        self.info_text = [
            "=== System Status ===",
            f"FPS: {tracking_info.get('fps', 0):.1f}",
            f"Recognition: {'ON' if self.recognition_enabled else 'OFF'}",
            f"Tracking: {'ON' if self.tracking_enabled else 'OFF'}",
            f"Motion: {'ON' if self.motion_enabled else 'OFF'}",
            "",
            "=== Recognition Stats ===",
            f"Status: {'Detecting' if not tracking_info.get('tracking', False) else 'Tracking'}",
            f"Object: {tracking_info.get('object_class', 'None')}",
            f"Confidence: {tracking_info.get('confidence', 0):.2f}",
            f"Stability: {stability_time:.1f}s",
            f"Detections: {tracking_info.get('consecutive_detections', 0)}",
            "",
            "=== Tracking Stats ===",
            f"Tracker: {tracking_info.get('tracker_type', 'N/A')}",
            f"Status: {'Active' if tracking_info.get('tracking', False) else 'Inactive'}"
        ]
        
        # Draw info text
        self._draw_info_text(display_frame)
        
        # Add keyboard shortcuts overlay
        self._add_keyboard_shortcuts(display_frame)
        
        # Show frame
        cv2.imshow(self.window_name, display_frame)
        
        # Return key press
        return cv2.waitKey(1) & 0xFF

    def _add_keyboard_shortcuts(self, frame):
        """Add keyboard shortcuts to the frame"""
        shortcuts = [
            "q - Quit",
            "r - Reset",
            f"i - Image Recognition: {'ON' if self.recognition_enabled else 'OFF'}",
            f"t - Image Tracking: {'ON' if self.tracking_enabled else 'OFF'}",
            f"m - Motion Detection: {'ON' if self.motion_enabled else 'OFF'}"
        ]
        
        # Calculate text dimensions
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 1
        line_spacing = 25
        
        # Find the longest text for width calculation
        max_width = 0
        for shortcut in shortcuts:
            (width, height), _ = cv2.getTextSize(shortcut, font, font_scale, font_thickness)
            max_width = max(max_width, width)
        
        # Calculate border dimensions with padding
        padding = 20
        border_width = max_width + padding * 2
        border_height = (len(shortcuts) - 1) * line_spacing + padding * 2
        
        # Position for keyboard shortcuts (bottom left)
        y_start = frame.shape[0] - border_height - 10  # 10 pixels from bottom
        x_start = 10  # 10 pixels from left
        
        # Draw border
        cv2.rectangle(frame, 
                     (x_start, y_start),
                     (x_start + border_width, y_start + border_height),
                     (255, 255, 255),  # White border
                     1)  # Border thickness
        
        # Add each shortcut with left-justified text
        for i, shortcut in enumerate(shortcuts):
            y = y_start + padding + i * line_spacing + 5
            # Calculate text position to left-justify it
            x = x_start + padding
            cv2.putText(frame,
                       shortcut,
                       (x, y),
                       font,
                       font_scale,
                       (255, 255, 255),
                       font_thickness)

    def cleanup(self):
        """Clean up UI resources"""
        cv2.destroyWindow(self.window_name) 