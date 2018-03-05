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



def check_label():
	root = 'data'
	# videos = os.listdir(root)
	videos = ['data_20180109']
	print("total ", len(videos))
	for video in videos:
		sub_video_path = os.path.join(root, video)
		sub_videos = os.listdir(sub_video_path)
		
		for i in sub_videos:
			path =os.path.join(sub_video_path,i)
			
			items = os.listdir(path)
			image = [e for e in items if not e.endswith('.txt')]
			with open(os.path.join(path, 'labels.txt')) as f:
				labels = f.read()
			# print(len(labels), len(image))
			if len(labels) != len(image):
				print(path)

def write_label_to_image():
	root = 'data'
	# videos = os.listdir(root)
	videos = ['data_20180109']
	font = cv2.FONT_HERSHEY_SIMPLEX
	print("total ", len(videos))
	for video in videos:
		sub_video_path = os.path.join(root, video)
		sub_videos = os.listdir(sub_video_path)
		sub_videos = [e for e in sub_videos if not e.endswith('_result')]
		for i in sub_videos:
			path =os.path.join(sub_video_path,i)
			
			items = os.listdir(path)
			image = [e for e in items if not e.endswith('.txt')]
			with open(os.path.join(path, 'labels.txt')) as f:
				labels = f.read()
			# print(len(labels), len(image))
			if len(labels) != len(image):
				print(path)
				continue
			new_path = os.path.join(sub_video_path, i+'_result')
			if os.path.exists(new_path):
				shutil.rmtree(new_path)
			os.mkdir(new_path)

			for j in range(len(labels)):
				new_img = cv2.imread(os.path.join(path, str(j)+'.png'))
				cv2.putText(new_img, labels[j], (30, 50), font, 2, (255,255,0),5)
				cv2.imwrite(os.path.join(new_path, str(j)+'.png'), new_img)




if __name__ == '__main__':

    ######0表示翻页，1表示注视拍摄
    # predict_video(predict(args), args.video_path)

    # check_label()
    write_label_to_image()

