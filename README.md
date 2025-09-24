# animal-behaviour-alert
Duck behavior detection &amp; lightweight species alert (YOLO-lite). PyTorch/Ultralytics, realtime inference, clean demos.


# Animal Behaviour & Species Alert (YOLO-lite)

Two compact computer-vision projects:
- **Duck Behaviour Detection** – detect & classify duck behaviours from video.
- **Species Alert (YOLO-lite)** – lightweight species detection with real-time alerts.

## Quickstart
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

## Inference 
python species_alert_yolo_lite/infer.py --source path/to/video.mp4 --weights weights/best.pt
python duck_behavior_detection/infer.py --source path/to/video.mp4 --weights weights/best.pt

## Data & Weights
This repo excludes large datasets/weights. Put tiny samples in datasets/ and download full weights from the Releases page or the link provided in each sub-README.

## structure
duck_behavior_detection/   # behaviour classes, loaders, train/infer scripts
species_alert_yolo_lite/   # YOLO-lite configs + train/infer
datasets/                  # empty by default (add small samples only)
docs/                      # figures/gifs for the README
dev-docs/                  # development documentation (non-code)
requirements.txt

## Contact
Haoyue (Howie) Zhang · haz143@ucsd.edu|jackalgorithman@qq.com



