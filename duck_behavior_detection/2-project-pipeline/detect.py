#!usr/bin/env python
# encoding:utf-8
from __future__ import division




"""
Function: Object detection inference calculation module
"""



import os
import sys
import cv2
import time
import argparse
import random
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.dataloaders import letterbox
from utils.plots import plot_one_box
import matplotlib.pyplot as plt
from PIL import ImageTk, Image




# configuration
names_list = ["duck"]
COLORS = np.random.randint(0, 255, size=(len(names_list), 3), dtype="uint8")
# Model Path
model_path = "runs/train/yolov5s/weights/best.pt"






def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    """
    Visualization of results
    """
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )
    return img




def inference(image_path):
    """
    Inference calculation
    """
    model = attempt_load(model_path, device="cpu")
    print("Loading Image From: ", image_path)
    img = cv2.imread(image_path)
    showimg = img
    with torch.no_grad():
        img = letterbox(img, new_shape=640)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to("cpu")
        img = img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred = model(img)[0]
        res_list = []
        pred = non_max_suppression(
            pred,
            0.25,
            0.45,
        )
        print("pred: ", pred)
        for i, det in enumerate(pred):
            if det is not None and len(det):
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], showimg.shape
                ).round()
                for *xyxy, conf, cls in reversed(det):
                    name = names_list[int(cls)]
                    color = [int(c) for c in COLORS[int(cls)]]
                    label = name + ":" + str(round(float(conf), 3))
                    print("label info: ", label)
                    res_list.append(
                        [name, round(float(conf), 3), [int(O) for O in xyxy]]
                    )
    #cv2.imwrite("prediction.jpg", showimg)
    return Image.fromarray(cv2.cvtColor(showimg, cv2.COLOR_BGR2RGB)), res_list


