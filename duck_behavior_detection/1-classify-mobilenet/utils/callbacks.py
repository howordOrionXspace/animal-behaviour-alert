#!usr/bin/env python
# encoding:utf-8
from __future__ import division


"""
功能：   CallBacks模块
"""




import os
import json
import time
import keras
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import scipy.signal
from keras import backend as K
import math




class LossHistory(keras.callbacks.Callback):
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.losses = []
        self.val_loss = []
        self.acc = []
        self.val_acc = []
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)


    def on_epoch_end(self, epoch, logs={}):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        # loss
        self.losses.append(logs.get("loss"))
        self.val_loss.append(logs.get("val_loss"))
        with open(os.path.join(self.log_dir, "epoch_loss.txt"), "a") as f:
            f.write(str(logs.get("loss")))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), "a") as f:
            f.write(str(logs.get("val_loss")))
            f.write("\n")
        self.loss_plot()
        # acc
        self.acc.append(logs.get("categorical_accuracy"))
        self.val_acc.append(logs.get("val_categorical_accuracy"))
        with open(os.path.join(self.log_dir, "epoch_acc.txt"), "a") as f:
            f.write(str(logs.get("categorical_accuracy")))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_acc.txt"), "a") as f:
            f.write(str(logs.get("val_categorical_accuracy")))
            f.write("\n")
        self.acc_plot()



    def loss_plot(self):
        iters = range(len(self.losses))
        plt.figure()
        plt.plot(iters, self.losses, "red", linewidth=2, label="train loss")
        plt.plot(iters, self.val_loss, "coral", linewidth=2, label="val loss")
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15
            plt.plot(
                iters,
                scipy.signal.savgol_filter(self.losses, num, 3),
                "green",
                linestyle="--",
                linewidth=2,
                label="smooth train loss",
            )
            plt.plot(
                iters,
                scipy.signal.savgol_filter(self.val_loss, num, 3),
                "#8B4513",
                linestyle="--",
                linewidth=2,
                label="smooth val loss",
            )
        except:
            pass
        plt.grid(True)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("A Loss Curve")
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))
        plt.cla()
        plt.close("all")



    def acc_plot(self):
        iters = range(len(self.acc))
        plt.figure()
        plt.plot(iters, self.acc, "red", linewidth=2, label="train accuracy")
        plt.plot(iters, self.val_acc, "coral", linewidth=2, label="val accuracy")
        try:
            if len(self.acc) < 25:
                num = 5
            else:
                num = 15
            plt.plot(
                iters,
                scipy.signal.savgol_filter(self.acc, num, 3),
                "green",
                linestyle="--",
                linewidth=2,
                label="smooth train accuracy",
            )
            plt.plot(
                iters,
                scipy.signal.savgol_filter(self.val_acc, num, 3),
                "#8B4513",
                linestyle="--",
                linewidth=2,
                label="smooth val accuracy",
            )
        except:
            pass
        plt.grid(True)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("A Accuracy Curve")
        plt.legend(loc="upper right")
        plt.savefig(os.path.join(self.log_dir, "epoch_acc.png"))
        plt.cla()
        plt.close("all")



class ExponentDecayScheduler(keras.callbacks.Callback):


    def __init__(self, decay_rate, verbose=0):
        super(ExponentDecayScheduler, self).__init__()
        self.decay_rate = decay_rate
        self.verbose = verbose
        self.learning_rates = []


    def on_epoch_end(self, batch, logs=None):
        learning_rate = K.get_value(self.model.optimizer.lr) * self.decay_rate
        K.set_value(self.model.optimizer.lr, learning_rate)
        if self.verbose > 0:
            print("Setting learning rate to %s." % (learning_rate))


class WarmUpCosineDecayScheduler(keras.callbacks.Callback):


    def __init__(self, T_max, eta_min=0, verbose=0):
        super(WarmUpCosineDecayScheduler, self).__init__()
        self.T_max = T_max
        self.eta_min = eta_min
        self.verbose = verbose
        self.init_lr = 0
        self.last_epoch = 0


    def on_train_begin(self, batch, logs=None):
        self.init_lr = K.get_value(self.model.optimizer.lr)


    def on_epoch_end(self, batch, logs=None):
        learning_rate = (
            self.eta_min
            + (self.init_lr - self.eta_min)
            * (1 + math.cos(math.pi * self.last_epoch / self.T_max))
            / 2
        )
        self.last_epoch += 1
        K.set_value(self.model.optimizer.lr, learning_rate)
        if self.verbose > 0:
            print("Setting learning rate to %s." % (learning_rate))

