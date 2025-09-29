#!usr/bin/env python
# encoding:utf-8
from __future__ import division




"""
Function: Common component module

Provide common tools for implementation 
"""





import os
import time
import json
import math
from functools import partial
import numpy as np
from PIL import Image





def letterbox_image(image, size):
    """
    Resize without distortion
    """
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new("RGB", size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image




def get_classes(classes_path):
    """
    Extract category list
    """
    with open(classes_path, encoding="utf-8") as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)




def cvtColor(image):
    """
    Convert image to RGB image
    """
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert("RGB")
        return image




def preprocess_input(x):
    """
    Preprocessing training images
    """
    x /= 127.5
    x -= 1.0
    return x

