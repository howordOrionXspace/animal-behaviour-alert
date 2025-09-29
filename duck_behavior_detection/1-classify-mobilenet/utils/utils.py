#!usr/bin/env python
# encoding:utf-8
from __future__ import division



"""
功能：   公共组件模块
        提供公用工具实现
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
    不失真的resize
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
    提取类别清单
    """
    with open(classes_path, encoding="utf-8") as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)




def cvtColor(image):
    """
    图像转换成RGB图像
    """
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert("RGB")
        return image




def preprocess_input(x):
    """
    预处理训练图片
    """
    x /= 127.5
    x -= 1.0
    return x




def get_lr_scheduler(
    lr_decay_type,
    lr,
    min_lr,
    total_iters,
    warmup_iters_ratio=0.1,
    warmup_lr_ratio=0.1,
    no_aug_iter_ratio=0.3,
    step_num=10,
):

    def warm_cos_lr(
        lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters
    ):
        if iters <= warmup_total_iters:
            lr = (lr - warmup_lr_start) * pow(
                iters / float(warmup_total_iters), 2
            ) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0
                + math.cos(
                    math.pi
                    * (iters - warmup_total_iters)
                    / (total_iters - warmup_total_iters - no_aug_iter)
                )
            )
        return lr


    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n = iters // step_size
        out_lr = lr * decay_rate**n
        return out_lr



    if lr_decay_type == "cos":
        warmup_total_iters = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(
            warm_cos_lr,
            lr,
            min_lr,
            total_iters,
            warmup_total_iters,
            warmup_lr_start,
            no_aug_iter,
        )
    else:
        decay_rate = (min_lr / lr) ** (1 / (step_num - 1))
        step_size = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)
    return func

