# Real-Time Object Detection and Tracking

This project implements real-time object detection and tracking using YOLOv8 for detection and OpenCV's tracking algorithms for continuous object tracking.

## Features

- Real-time object detection using YOLOv8
- Multiple tracking algorithms (CSRT, KCF, MOSSE)
- Frame preprocessing for improved performance
- FPS display
- Object class and confidence display
- Interactive controls

## Requirements

- Python 3.8+
- OpenCV
- PyTorch
- Ultralytics YOLO
- NumPy

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main script:
```bash
python main.py
```

### Controls

- Press 'q' to quit the application
- Press 'r' to reset tracking (useful when tracking is lost or you want to track a different object)

### Display Information

The application shows:
- Bounding box around the tracked object
- Object class name
- Detection confidence
- Tracker type
- Current FPS

## Available Trackers

The application supports multiple tracking algorithms from OpenCV, each with its own characteristics:

1. **CSRT** (Discriminative Correlation Filter with Channel and Spatial Reliability)
   - Good balance of speed and accuracy
   - Better handling of scale changes
   - More accurate but slower than KCF

2. **KCF** (Kernelized Correlation Filter)
   - Fast and accurate
   - Good for real-time tracking
   - Less accurate with scale changes

3. **MOSSE** (Minimum Output Sum of Squared Error)
   - Very fast
   - Less accurate
   - Good for simple tracking tasks

4. **MIL** (Multiple Instance Learning)
   - Robust to occlusions
   - Moderate speed
   - Good for complex scenes

5. **BOOSTING** (AdaBoost classifier)
   - Traditional algorithm
   - Slower but robust
   - Good for simple tracking

6. **MEDIANFLOW** (Median Flow tracker)
   - Good for slow-moving objects
   - Handles scale changes well
   - Moderate speed

7. **TLD** (Tracking, Learning and Detection)
   - Combines tracking and detection
   - Good for long-term tracking
   - Can recover from tracking failures
   - Slower than other algorithms

### Tracker Selection Guide

Choose a tracker based on your specific needs:
- For speed: MOSSE or KCF
- For accuracy: CSRT
- For robustness: MIL or TLD
- For scale changes: CSRT or MEDIANFLOW
- For long-term tracking: TLD

To change the tracker, modify the `TRACKER_TYPE` in `config/settings.py`.

### Detailed Tracker Analysis

#### CSRT
**Strengths:**
- High accuracy in tracking
- Good handling of scale changes
- Robust to partial occlusions
- Works well with complex backgrounds

**Weaknesses:**
- Slower than other trackers
- Higher computational requirements
- May struggle with very fast movements
- Memory intensive

#### KCF
**Strengths:**
- Fast execution speed
- Good accuracy for real-time applications
- Efficient with computational resources
- Works well with consistent lighting

**Weaknesses:**
- Poor handling of scale changes
- Can lose track during occlusions
- Sensitive to fast movements
- May drift with similar-looking objects

#### MOSSE
**Strengths:**
- Extremely fast execution
- Low computational requirements
- Good for simple tracking scenarios
- Works well with consistent objects

**Weaknesses:**
- Lower accuracy compared to other trackers
- Poor handling of scale changes
- Sensitive to occlusions
- Can drift with similar objects

#### MIL
**Strengths:**
- Robust to occlusions
- Good at handling appearance changes
- Works well in complex scenes
- Can handle partial object visibility

**Weaknesses:**
- Moderate speed
- May struggle with fast movements
- Can be sensitive to initialization
- Higher memory usage

#### BOOSTING
**Strengths:**
- Robust to appearance changes
- Good for simple tracking scenarios
- Can handle partial occlusions
- Works well with consistent objects

**Weaknesses:**
- Slower than modern trackers
- Higher computational requirements
- May struggle with fast movements
- Can drift with similar objects

#### MEDIANFLOW
**Strengths:**
- Good handling of scale changes
- Works well with slow-moving objects
- Robust to small occlusions
- Good for tracking rigid objects

**Weaknesses:**
- Poor performance with fast movements
- Can lose track during large occlusions
- May struggle with non-rigid objects
- Sensitive to initialization

#### TLD
**Strengths:**
- Can recover from tracking failures
- Good for long-term tracking
- Combines tracking and detection
- Works well with occlusions

**Weaknesses:**
- Slower than other trackers
- Higher computational requirements
- Complex implementation
- May have false positives

## Configuration

Adjust settings in `config/settings.py`:
- Video source
- Frame dimensions
- Tracker type
- Confidence threshold
- Preprocessing options

## Project Structure

```
.
├── config/
│   └── settings.py
├── src/
│   ├── detector.py
│   └── tracker.py
├── utils/
│   ├── preprocessing.py
│   └── video_utils.py
├── main.py
├── requirements.txt
└── README.md
```

## License

[Your License] 