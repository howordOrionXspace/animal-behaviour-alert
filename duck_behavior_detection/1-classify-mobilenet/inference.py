#!usr/bin/env python
# encoding:utf-8
from __future__ import division




"""
Function: Inference module 
"""




import os
import time
import json
from PIL import Image
from recognition import *




myModel = recognitionModel()




while True:
    inputImg = input("Please Input Image Path[autoStart Afer Enter]: ")
    print("Loading Image From: ", inputImg)
    start = time.time()
    nowImg = Image.open(inputImg)
    pred_label, pred_proba = myModel.recognitionImage(nowImg)
    print("pred_label: ", pred_label)
    print("pred_proba: ", pred_proba)
    end = time.time()
    delta = end - start
    print("delta: ", delta)
    print("\n")
