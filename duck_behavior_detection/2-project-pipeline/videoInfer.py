#!usr/bin/env python
# encoding:utf-8
from __future__ import division




"""
Function: Video inference calculation module
"""





import os
import cv2
import time
import json
import detect
from classify import *












# configuration
names_list = ['Limping', 'LyingDown', 'Sitting', 'SpreadingWings', 'Standing', 'Walking']
# colour
COLORS = np.random.randint(0, 255, size=(len(names_list), 3), dtype="uint8")





def cutBoxCV(img, box):
    """
    slicing
    """
    x1, y1, x2, y2 = box
    res = img[y1:y2, x1:x2]
    return res




def plotInfo(box, img, color=None, label=None, line_thickness=3):
    """
    Visualization
    """
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
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




class CameraInference(object):
    def __init__(self, modelDir="runs/train/", delta=10, camera=0):
        """
        initialization
        """
        self.modelDir = modelDir
        self.delta = delta
        self.camera = camera
        self.model = recognitionModel()


    def inference(self):
        """
        Overall inference calculation
        """
        self.cameraCapture = cv2.VideoCapture(self.camera)
        success, frame = self.cameraCapture.read()
        fps=int(self.cameraCapture.get(cv2.CAP_PROP_FPS))
        frames=int(self.cameraCapture.get(cv2.CAP_PROP_FRAME_COUNT))
        print("fps: ", fps)
        print("frames: ", frames)
        time.sleep(5)
        counter = 0
        while success and counter <= frames:
            counter += 1
            print("counter: ", counter)
            if counter % self.delta == 0:
                try:
                    success, frame = self.cameraCapture.read()
                    if frame is None:
                        break
                    if success:
                        cv2.imwrite("temp.jpg", frame)
                        # detection
                        resImg, resData = detect.inference("temp.jpg")
                        for one_list in resData:
                            one_obj, one_conf, one_box = one_list
                            one_img = cutBoxCV(frame, one_box)
                            one_img = Image.fromarray(
                                cv2.cvtColor(one_img, cv2.COLOR_BGR2RGB)
                            )
                            # identify
                            one_label, one_proba = self.model.recognitionImage(one_img)
                            label_info = "Action: "+ one_label +", Probality: "+str(one_proba)
                            color_ind = names_list.index(one_label)
                            one_color = [int(c) for c in COLORS[color_ind]]
                            frame = plotInfo(
                                one_box,
                                frame,
                                label=label_info,
                                color=one_color,
                                line_thickness=2,
                            )
                        os.remove("temp.jpg")
                    else:
                        break
                except:
                    pass
            cv2.imshow("Camera", frame)
            if cv2.waitKey(10) & 0xFF == 27:
                break
        self.cameraCapture.release()
        cv2.destroyAllWindows()






if __name__ == "__main__":
    camera = CameraInference(camera="test.mp4")
    camera.inference()
