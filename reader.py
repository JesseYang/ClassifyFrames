import os, sys
import cv2
import pickle
import numpy as np
from scipy import misc
import struct
import six
from six.moves import urllib, range
import copy
import logging
from PIL import Image
from tensorpack import *
import random
import uuid
from PIL import Image,ImageDraw,ImageFont
try:
    from .cfgs.config import cfg
except Exception:
    from cfgs.config import cfg

#from morph import warp

def get_center_frame_list(fname):
    with open(fname) as f:
        content = f.readlines()
    content = [ele.strip() for ele in content]
    return content

def read_data(content):
    center_frame_path, label = content.split(' ')
    label = int(label)
    dir_path = '/'.join(center_frame_path.split('/')[:-1])
    center_frame_idx = int(center_frame_path.split('/')[-1].split('.')[0])
    num_frames = len(os.listdir(dir_path)) - 1
    margin = len(cfg.frame_extract_pattern) // 2
    if center_frame_idx - margin < 0 or center_frame_idx + margin >= num_frames:
        return

    frames_path = [os.path.join(dir_path, '{}.png'.format(frame_idx)) for frame_idx in range(center_frame_idx - margin, center_frame_idx + margin + 1)]
    frames = []
    angle = np.random.choice(cfg.angle, p=[0.25, 0.25, 0.25, 0.25])
   
    for i in range(len(cfg.frame_extract_pattern)):
        if cfg.frame_extract_pattern[i]:
            # frames.append(misc.imread(frames_path[i], mode = 'L'))
            img = Image.open(frames_path[i])
            img = img.convert("L")

            img = img.rotate(angle)
            frames.append(np.array(img))
            # misc.imsave(os.path.join("test_dir", "or"+str(i)+"_"+str(uuid.uuid4())+".jpg"), np.array(img))
    frames_stack = np.asarray(frames)
    # print(frames_stack.shape)
    # d = ImageDraw.Draw(blank)
    # print(blank.shape)
    # misc.imsave(os.path.join("test_dir", str(uuid.uuid4())+".jpg"), np.asarray(blank))
    # print(np.asarray(d).shape)
    if random.uniform(0, 1) >= 0.5:
        mean_value = np.mean(frames_stack)
        blank = np.zeros((cfg.w, cfg.h)) + mean_value
        scale_w = random.uniform(0.3, 1)
        scale_h = random.uniform(0.3, 1)
        while ((scale_w / scale_h) < 0.5 or (scale_w / scale_h) > 2):
            scale_w = random.uniform(0.3, 1)
            scale_h = random.uniform(0.3, 1)
        new_w = int(np.round(scale_w * cfg.w))
        new_h = int(np.round(scale_h * cfg.h))
        coor_x = random.randint(0, cfg.w - new_w)
        coor_y = random.randint(0, cfg.h - new_h)

        new_frames = []
        for i in range(len(cfg.frame_extract_pattern)):
            blank[coor_y:coor_y + new_h, coor_x:coor_x + new_w] = cv2.resize(frames_stack[i], (new_w, new_h))
            new_frames.append(np.array(blank))
            # misc.imsave(os.path.join("test_dir", "v"+str(i)+"_"+str(uuid.uuid4())+".jpg"), np.asarray(blank))
        frames_stack = np.asarray(new_frames)
        # print(frames_stack.shape)
    


    frames_stack = frames_stack.swapaxes(0,2)
    # print(frames_stack[0])
    return [frames_stack, label]

class Data(RNGDataFlow):
    def __init__(self, train_or_test):
        assert train_or_test in ['train', 'test']
        fname_list = cfg.train_list if train_or_test == 'train' else cfg.test_list
        self.shuffle = (train_or_test == 'train')
        self.center_frame_list = []
        for fname in fname_list:
            self.center_frame_list.extend(get_center_frame_list(fname))
    
    def size(self):
        return int(len(self.center_frame_list) * 0.9)

    def get_data(self):
        idxs = np.arange(len(self.center_frame_list))
        if self.shuffle:
            self.rng.shuffle(idxs)
        for k in idxs:
            frame_content = self.center_frame_list[k]
            res = read_data(frame_content)
            if res:
                yield res

if __name__ == '__main__':
    ds = Data('train')
    ds.reset_state()
    # generator = ds.get_data()
    # for i in generator:
    #     print(i[0].shape, i[1])
    count = 0
    while count < 1:
        ps = ds.get_data()
        n=next(ps)
        count += 1
