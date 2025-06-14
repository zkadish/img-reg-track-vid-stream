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