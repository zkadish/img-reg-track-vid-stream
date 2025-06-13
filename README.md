# Image Recognition and Tracking from Video Stream

This project implements real-time image recognition and tracking from a video stream using computer vision techniques.

## Features
- Real-time video stream processing
- Image recognition and tracking
- Configurable tracking parameters
- Support for multiple tracking methods

## Setup
1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python main.py
```

## Project Structure
- `src/`: Core functionality
  - `tracker.py`: Image tracking implementation
  - `detector.py`: Image detection implementation
- `utils/`: Helper functions
  - `video_utils.py`: Video stream handling
  - `image_utils.py`: Image processing utilities
- `config/`: Configuration files
  - `settings.py`: Application settings
- `main.py`: Main application entry point

## Configuration
Edit `config/settings.py` to modify tracking parameters and video source settings. 