#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import shutil
import ntpath
import numpy as np
import math
from scipy import misc
import argparse
import json
import cv2
import random
from tensorpack.tfutils.sesscreate import SessionCreatorAdapter, NewSessionCreator
from tensorflow.python import debug as tf_debug
import uuid
import tensorflow as tf
from tensorpack import *
from cfgs.config import cfg
import time

try:
    from .train import Model
    from .cfgs.config import cfg
except Exception:
    from train import Model
    from cfgs.config import cfg






def predict_video(predict_func, video_path):
    assert os.path.exists(video_path), ' video_path file not exists'
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened == False:
        cap.open(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_counts = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))#total frames of this video
    print(video_path, " : ", frame_counts, 'frames')

    write_video_name = video_path.split("/")[-1]
    ####isColor=0
    videoWrite = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), cap.get(cv2.CAP_PROP_FPS), (width, height))
   
    total_frames = []
    images = []
    while cap.isOpened:
        ret, frame = cap.read()
        if ret == False:
            break
        images.append(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(frame, (cfg.w, cfg.h))
        total_frames.append(img)
    print("total frame ", len(total_frames))
    margin = len(cfg.frame_extract_pattern) // 2
 
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(len(total_frames)):
        if i % 100 == 0:
            print(i)
        
        if i - margin >= 0 and i + margin <= (frame_counts - 1):
            # print(i)
            # if i+margin == (frame_counts -1):
            #     batch_frames = total_frames[i - margin:]
            # else:
            batch_frames = [total_frames[frame_idx] for frame_idx in range(i - margin, i + margin+1)]
            # print(len(batch_frames))
            inputs = np.expand_dims((np.asarray(batch_frames).swapaxes(0,2)), axis=0)
            predicted_result = predict_func([inputs])[0][0]
            neg = round(predicted_result[0], 4)
            pos = round(predicted_result[1], 4)
            if pos >= 0.5:
            	label = '1'
            else:
            	label = '0'
            cv2.putText(images[i], label, (30, height-30), font, 1, (255,255,0),5)
            cv2.putText(images[i], str(pos), (width-100, height-30), font, 1, (255,233,134),5)
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        videoWrite.write(images[i])


def predict(args):
    sess_init = SaverRestore(args.model)
    model = Model()
    predict_config = PredictConfig(session_init=sess_init,
                                   model=model,
                                   input_names=['input'],
                                   output_names=['output'])
    
    predict_func = OfflinePredictor(predict_config)

    return predict_func

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='path to the model file', required=True)
    parser.add_argument('--video_path', help='path to the input video_path', default='test_video.mp4')
    args = parser.parse_args()
    ######0表示翻页，1表示注视拍摄
    predict_video(predict(args), args.video_path)

