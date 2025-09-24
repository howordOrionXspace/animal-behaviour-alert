# animal-behaviour-alert
Duck behavior detection &amp; lightweight species alert (YOLO-lite). PyTorch/Ultralytics, realtime inference, clean demos.


# Animal Behaviour & Species Alert (YOLO-lite)

Two compact computer-vision projects:
- **Duck Behaviour Detection** – detect & classify duck behaviours from video.
- **Species Alert (YOLO-lite)** – lightweight species detection with real-time alerts.

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt


python species_alert_yolo_lite/infer.py --source path/to/video.mp4 --weights weights/best.pt
python duck_behavior_detection/infer.py --source path/to/video.mp4 --weights weights/best.pt

