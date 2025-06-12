# Object Detection Efficiency Testing

This repository contains scripts for evaluating and comparing the efficiency and accuracy of YOLO object detection models across different configurations and inference conditions. The focus is on understanding trade-offs between model size, speed, and detection reliability in constrained or real-time settings.

## What This Project Does

This project explores performance differences between:
- **YOLOv8 Large** and **YOLOv8 Nano** models
- **Standard inference** and **diagnostic-enhanced modes**

Each file is structured to isolate testing scenarios:
- `ObjectDetectLarge.py`: YOLOv8 Large standard inference
- `ObjectDetectLargeDiag.py`: YOLOv8 Large with diagnostic logging
- `ObjectDetectNano.py`: YOLOv8 Nano standard inference
- `ObjectDetectNanoDiag.py`: YOLOv8 Nano with diagnostic logging

The diagnostic versions include detailed breakdowns of:
- FPS and latency metrics
- Frame-by-frame detection consistency
- Hardware usage insights (GPU/CPU load)

## Why This Matters

In real-world applications like drones, robotics, and autonomous navigation, model selection involves balancing precision and computational efficiency. This project helps benchmark those trade-offs to inform deployment decisions under resource constraints.

## How It Works

Each script:
1. Loads a pre-trained YOLOv8 model.
2. Captures video input (via webcam or file).
3. Runs inference and tracks performance metrics.
4. Logs or visualizes output depending on mode.

## Use Cases

- Rapid prototyping for embedded systems
- Benchmarking detection performance on different GPUs
- Educational tool to understand model scaling
- Debugging and system diagnostics for CV pipelines

## File Overview

| File                    | Description                                      |
|-------------------------|--------------------------------------------------|
| `ObjectDetectLarge.py`  | YOLOv8 Large model basic detection script        |
| `ObjectDetectLargeDiag.py` | Large model with extended diagnostics         |
| `ObjectDetectNano.py`   | Lightweight YOLOv8 Nano detection script         |
| `ObjectDetectNanoDiag.py` | Nano model with diagnostic insights           |

## System Requirements

- Python 3.8+
- OpenCV
- Ultralytics YOLOv8
- PyTorch with CUDA (for GPU use)

Install dependencies:
```bash
pip install -r requirements.txt
