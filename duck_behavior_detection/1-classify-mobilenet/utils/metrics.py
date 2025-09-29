#!usr/bin/env python
# encoding:utf-8
from __future__ import division


"""
功能：   模型计算评估模块
"""




import os
import csv
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import *
from sklearn.model_selection import *



def four(true_list, pred_list):
    """
    指标
    """
    res_dict = {}
    accuracy = accuracy_score(true_list, pred_list)
    precision = precision_score(true_list, pred_list, average="macro")
    recall = recall_score(true_list, pred_list, average="macro")
    f1 = f1_score(true_list, pred_list, average="macro")
    res_dict["accuracy"] = accuracy
    res_dict["precision"] = precision
    res_dict["recall"] = recall
    res_dict["f1"] = f1
    print("res_dict: ", res_dict)
    return res_dict


def evaluteModel(regModel, lines, resDir):
    correct_1 = 0
    correct_5 = 0
    preds = []
    labels = []
    total = len(lines)
    true_list, pred_list = [],[]
    for index, line in enumerate(lines):
        annotation_path = line[1]
        x = Image.open(annotation_path)
        y = int(line[0])
        pred = regModel.imageInfer(x)
        pred_1 = np.argmax(pred)
        correct_1 += pred_1 == y
        pred_5 = np.argsort(pred)[::-1]
        pred_5 = pred_5[:5]
        correct_5 += y in pred_5
        preds.append(pred_1)
        labels.append(y)
        true_list.append(y)
        pred_list.append(pred_1)
        if index % 100 == 0:
            print("[%d/%d]" % (index, total))
    print("preds: ", preds)
    print("labels: ", labels)
    hist = fast_hist(np.array(labels), np.array(preds), len(regModel.class_names))
    print("hist: ", hist)
    Recall = per_class_Recall(hist)
    Precision = per_class_Precision(hist)
    print("Recall: ", Recall)
    print("Precision: ", Precision)
    show_results(resDir, hist, Recall, Precision, regModel.class_names)
    # F1
    F1 = per_class_F1(Recall, Precision)
    print("F1: ", F1)
    draw_plot_func(
        F1,
        regModel.class_names,
        "F1 = {0:.2f}%".format(np.nanmean(F1) * 100),
        "F1",
        resDir + "F1.png",
        tick_font_size=12,
        plt_show=False,
    )
    # Accuracy
    matrix = hist.tolist()
    Accuracy = []
    for i in range(len(matrix)):
        one_value = matrix[i][i]
        one_sum = sum(matrix[i])
        one_acc = one_value / one_sum
        Accuracy.append(one_acc)
    print("Accuracy: ", Accuracy)
    draw_plot_func(
        Accuracy,
        regModel.class_names,
        "Accuracy = {0:.2f}%".format(np.nanmean(Accuracy) * 100),
        "Accuracy",
        resDir + "Accuracy.png",
        tick_font_size=12,
        plt_show=False,
    )
    four_dict=four(true_list, pred_list)
    with open("metrics.json","w") as f:
        f.write(json.dumps(four_dict)) 
    return Accuracy, Precision, Recall, F1




def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)




def per_class_Recall(hist):
    return np.diag(hist) / np.maximum(hist.sum(1), 1)




def per_class_Precision(hist):
    return np.diag(hist) / np.maximum(hist.sum(0), 1)




def per_class_F1(P, R):
    R = R.tolist()
    P = P.tolist()
    print("P: ", P)
    print("R: ", R)
    F1 = []
    for i in range(len(R)):
        one_F1 = 2 * (P[i] * R[i]) / (P[i] + R[i])
        F1.append(one_F1)
    return F1




def adjust_axes(r, t, fig, axes):
    bb = t.get_window_extent(renderer=r)
    text_width_inches = bb.width / fig.dpi
    current_fig_width = fig.get_figwidth()
    new_fig_width = current_fig_width + text_width_inches
    propotion = new_fig_width / current_fig_width
    x_lim = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1] * propotion])




def draw_plot_func(
    values,
    name_classes,
    plot_title,
    x_label,
    output_path,
    tick_font_size=12,
    plt_show=True,
):
    fig = plt.gcf()
    axes = plt.gca()
    plt.barh(range(len(values)), values, color="royalblue")
    plt.title(plot_title, fontsize=tick_font_size + 2)
    plt.xlabel(x_label, fontsize=tick_font_size)
    plt.yticks(range(len(values)), name_classes, fontsize=tick_font_size)
    r = fig.canvas.get_renderer()
    for i, val in enumerate(values):
        str_val = " " + str(val)
        if val < 1.0:
            str_val = " {0:.2f}".format(val)
        t = plt.text(val, i, str_val, color="royalblue", va="center", fontweight="bold")
        if i == (len(values) - 1):
            adjust_axes(r, t, fig, axes)

    fig.tight_layout()
    fig.savefig(output_path)
    if plt_show:
        plt.show()
    plt.close()




def show_results(
    miou_out_path, hist, Recall, Precision, name_classes, tick_font_size=12
):
    draw_plot_func(
        Recall,
        name_classes,
        "mRecall = {0:.2f}%".format(np.nanmean(Recall) * 100),
        "Recall",
        os.path.join(miou_out_path, "Recall.png"),
        tick_font_size=tick_font_size,
        plt_show=False,
    )
    print("Save Recall out to " + os.path.join(miou_out_path, "Recall.png"))

    draw_plot_func(
        Precision,
        name_classes,
        "mPrecision = {0:.2f}%".format(np.nanmean(Precision) * 100),
        "Precision",
        os.path.join(miou_out_path, "Precision.png"),
        tick_font_size=tick_font_size,
        plt_show=False,
    )
    print("Save Precision out to " + os.path.join(miou_out_path, "Precision.png"))

    with open(
        os.path.join(miou_out_path, "confusion_matrix.csv"), "w", newline=""
    ) as f:
        writer = csv.writer(f)
        writer_list = []
        writer_list.append([" "] + [str(c) for c in name_classes])
        for i in range(len(hist)):
            writer_list.append([name_classes[i]] + [str(x) for x in hist[i]])
        writer.writerows(writer_list)
    print(
        "Save confusion_matrix out to "
        + os.path.join(miou_out_path, "confusion_matrix.csv")
    )


