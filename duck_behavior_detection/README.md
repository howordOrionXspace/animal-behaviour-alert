# Duck Behaviour Detection

This module contains three parts:
1. **0-detect-yolov5** – object detector to find ducks in frames/videos.
2. **1-classify-mobilenet** – image classifier to label behaviour (e.g., feeding, preening).
3. **2-project-pipeline** – script that chains detection → crop/track → behaviour classification → timeline CSV/overlay video.

## Quickstart

python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r ../../requirements.txt

python 2-project-pipeline/pipeline.py \
  --source path/to/video.mp4 \
  --det-weights weights/duck_yolov5.pt \
  --cls-weights weights/behaviour_mobilenet.pth \
  --out outputs/demo

### `duck_behavior_detection/0-detect-yolov5/README.md`

# 0 – Duck Detection (YOLOv5)
Detects ducks in images/videos; outputs boxes (and optional tracks) for the behaviour stage.

## Use (inference)
**If you use the original YOLOv5 repo style (`detect.py`):**
python detect.py --weights weights/duck_yolov5.pt --source path/to/video.mp4 \
  --conf 0.25 --img 640 --save-txt --save-conf --project runs/detect --name duck

python - <<'PY'
from ultralytics import YOLO
m = YOLO("weights/duck_yolov5.pt")
m.predict(source="path/to/video.mp4", conf=0.25, imgsz=640, save=True, project="runs/detect", name="duck")
PY

##Train (YOLO format)
python train.py --img 640 --batch 16 --epochs 100 \
  --data data/duck.yaml --weights yolov5s.pt --name duck-det


---

### `duck_behavior_detection/1-classify-mobilenet/README.md`

# 1 – Behaviour Classification (MobileNet)

Classifies cropped duck images into behaviours (e.g., feeding, swimming, preening).

## Data format


## Train

python train.py --data data/behaviour --epochs 50 --batch 64 \
  --model mobilenet_v3_small --img-size 224 --out runs/cls


python infer.py --weights weights/behaviour_mobilenet.pth \
  --source path/to/crops --out outputs/cls.csv

python export.py --weights weights/behaviour_mobilenet.pth --onnx outputs/behaviour.onnx


### `duck_behavior_detection/2-project-pipeline/README.md`

# 2 – End-to-End Pipeline
Chains **detector → crop/track → behaviour classifier**, then writes:
- annotated video with boxes + behaviour labels
- a CSV timeline: `frame, t_sec, id, behaviour, conf`

## Run
python pipeline.py \
  --source path/to/video.mp4 \
  --det-weights ../0-detect-yolov5/weights/duck_yolov5.pt \
  --cls-weights ../1-classify-mobilenet/weights/behaviour_mobilenet.pth \
  --out outputs/run1 --conf 0.25 --img 640 --fps 30



