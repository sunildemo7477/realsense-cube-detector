# RealSense Cube Detector

Modular computer vision pipeline for detecting/tracking blue cubes (or general objects) with Intel RealSense D435. Evolves from experimental scripts to OOP production prototype.

## Features
- **Color-based**: HSV contours + 4-vertex approx (2D/3D).
- **ML-based**: YOLOv8 for classes + bboxes.
- **3D**: Point cloud masking, 8-vertex cube reconstruction.
- **Tracking**: Persistent last-known position.
- **Viz**: OpenCV streams + Open3D point clouds.

## Setup
1. Install: `pip install -r requirements.txt`
2. (Optional) Download YOLOv8: Auto on first run.
3. Run: `python main.py --mode color --device realsense`

## Modes
- `--mode color`: HSV cube detection (w/ optional 3D).
- `--mode yolo`: General object detection.

## Architecture
- `detectors.py`: OOP detection classes.
- `processors.py`: RealSense/3D logic.
- Config-driven via `config.yaml`.

## Experiments
Original scripts archived in `experiments/` for reference.

## License
MIT. Contributions welcome!