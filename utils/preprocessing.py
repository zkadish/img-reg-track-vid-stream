import cv2
import numpy as np
from typing import Tuple, Optional

def preprocess_frame(
    frame: np.ndarray,
    target_size: Tuple[int, int] = (640, 480),
    normalize: bool = True,
    denoise: bool = True,
    enhance_contrast: bool = True
) -> np.ndarray:
    """
    Optimized preprocessing pipeline for video frames.
    
    Args:
        frame: Input frame
        target_size: Target size for resizing (width, height)
        normalize: Whether to normalize pixel values
        denoise: Whether to apply denoising
        enhance_contrast: Whether to enhance contrast
    
    Returns:
        Preprocessed frame
    """
    # Resize first to reduce computation for subsequent operations
    if frame.shape[:2] != target_size[::-1]:  # OpenCV uses (height, width)
        frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
    
    # Convert to float32 once for all operations
    if normalize or enhance_contrast:
        frame = frame.astype(np.float32) / 255.0
    
    # Apply CLAHE for contrast enhancement (more efficient than global contrast)
    if enhance_contrast:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply((l * 255).astype(np.uint8))
        l = l.astype(np.float32) / 255.0
        lab = cv2.merge([l, a, b])
        frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Apply fast denoising
    if denoise:
        frame = cv2.fastNlMeansDenoisingColored(
            (frame * 255).astype(np.uint8),
            None,
            h=10,
            hColor=10,
            templateWindowSize=7,
            searchWindowSize=21
        ).astype(np.float32) / 255.0
    
    # Convert back to uint8 if needed
    if normalize or enhance_contrast:
        frame = (frame * 255).astype(np.uint8)
    
    return frame

def apply_gaussian_blur(
    frame: np.ndarray,
    kernel_size: Tuple[int, int] = (5, 5),
    sigma: float = 0.0
) -> np.ndarray:
    """
    Apply Gaussian blur with optimized parameters.
    
    Args:
        frame: Input frame
        kernel_size: Size of Gaussian kernel
        sigma: Standard deviation for Gaussian kernel
    
    Returns:
        Blurred frame
    """
    # Use separable Gaussian filter for better performance
    return cv2.GaussianBlur(frame, kernel_size, sigma)

def enhance_edges(
    frame: np.ndarray,
    low_threshold: int = 50,
    high_threshold: int = 150
) -> np.ndarray:
    """
    Optimized edge enhancement using Canny edge detection.
    
    Args:
        frame: Input frame
        low_threshold: Lower threshold for Canny
        high_threshold: Higher threshold for Canny
    
    Returns:
        Frame with enhanced edges
    """
    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Canny edge detection
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    
    # Dilate edges slightly to make them more visible
    kernel = np.ones((3,3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Convert edges to 3 channels and combine with original frame
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(frame, 0.7, edges, 0.3, 0)

def adjust_brightness_contrast(
    frame: np.ndarray,
    brightness: float = 1.0,
    contrast: float = 1.0
) -> np.ndarray:
    """
    Optimized brightness and contrast adjustment.
    
    Args:
        frame: Input frame
        brightness: Brightness adjustment factor
        contrast: Contrast adjustment factor
    
    Returns:
        Adjusted frame
    """
    # Use look-up table for faster processing
    lut = np.clip(contrast * np.arange(256) + brightness * 255, 0, 255).astype(np.uint8)
    return cv2.LUT(frame, lut) 