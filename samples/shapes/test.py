#!/usr/bin/env python
# coding: utf-8

import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import skimage.io
import argparse

import yaml
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

parser = argparse.ArgumentParser(
    description='MaskRcnn for Pasonatech test')
parser.add_argument('--class_names', default=['BG', 'scrach'],
                    type=list, help="['BG', 'class1', 'class2',...]")
parser.add_argument('--filter_classs_names', default=['scrach'],
                    type=list, help="['class1', 'class2',...], 推論したいclass")
parser.add_argument('--scores_thresh', default=0.9,
                    type=float, help="threshsoldより大きい結果のみprint")
parser.add_argument('--mode', default=0,
                    type=int, help="mode: select the result which you want {"
                                   "mode = 0:save image with bbox,class_name,score and mask//"
                                   "mode = 1:save image with bbox,class_name and score//"
                                   "mode = 2:save image with class_name,score and mask//"
                                   "mode = 3:save mask with black background}")
args = parser.parse_args()

class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "shapes"
    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 3 shapes
    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256
    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8 * 6, 16 * 6, 32 * 6, 64 * 6, 128 * 6)  # anchor side in pixels
    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32
    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100
    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5

def predict(class_names = args.class_names,filter_classs_names=args.filter_classs_names,
            scores_thresh=args.scores_thresh,mode=args.mode):
    # Create models in training mode
    config = ShapesConfig()
    config.display()
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir=MODEL_DIR)
    model_path = model.find_last()

    # Load trained weights (fill in path to trained weights here)
    assert model_path != "", "Provide path to trained weights"
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)
    

    IMAGE_DIR = os.path.join(ROOT_DIR, "test_data")
    save_dir = os.path.join(IMAGE_DIR, "dec_result")
    if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
    file_names = next(os.walk(IMAGE_DIR))[2]
    i=1
    for file in file_names:
        image = skimage.io.imread(os.path.join(IMAGE_DIR, file))

    # Run detection
        results = model.detect([image], verbose=1)

    # Visualize results
        r = results[0]
        image_name = 'dec_%s_%s' % (i,r['scores'])
        print(r['scores'])
        i += 1
        #visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
        visualize.save_image(image, image_name, r['rois'], r['masks'],r['class_ids'],r['scores'],class_names,save_dir=save_dir,
                             filter_classs_names=filter_classs_names,scores_thresh=scores_thresh,mode=mode)
if __name__ =='__main__':
    predict()




