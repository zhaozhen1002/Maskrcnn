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
import argparse
import yaml
from PIL import Image
# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

parser = argparse.ArgumentParser(
    description='MaskRcnn for Pasonatech train')
parser.add_argument('--init_weight', default='coco', choices=['coco', 'last'],
                    type=str, help='coco or last')
parser.add_argument('--epoch', default=10,
                    type=int, help='epochs')
parser.add_argument('--STEPS_PER_EPOCH', default=50,
                    type=int, help='STEPS_PER_EPOCH')
parser.add_argument('--GPU_COUNT', default=1,
                    type=int, help='GPU_COUNT')
parser.add_argument('--IMAGES_PER_GPU', default=4,
                    type=int, help='IMAGES_PER_GPU')
parser.add_argument('--NUM_CLASSES', default=2,
                    type=int, help='background + num of class')
parser.add_argument('--CLASS_NAME', default=['scrach'],
                    type=list, help='annotation label list')

args = parser.parse_args()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

class ShapesConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "shapes"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = args.GPU_COUNT
    IMAGES_PER_GPU = args.IMAGES_PER_GPU

    # Number of classes (including background)
    NUM_CLASSES = args.NUM_CLASSES  # background + 1 class

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
    STEPS_PER_EPOCH = args.STEPS_PER_EPOCH

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5

class DrugDataset(utils.Dataset):
    # get the num of object in IMAGE
    def get_obj_index(self, image):
        n = np.max(image)
        return n

    # get the label from yaml file
    def from_yaml_get_class(self, image_id):
        info = self.image_info[image_id]
        with open(info['yaml_path']) as f:
            temp = yaml.load(f.read())
            labels = temp['label_names']
            del labels[0]
        return labels

    # write draw_mask
    def draw_mask(self, num_obj, mask, image, image_id):
        # print("draw_mask-->",image_id)
        # print("self.image_info",self.image_info)
        info = self.image_info[image_id]
        # print("info-->",info)
        # print("info[width]----->",info['width'],"-info[height]--->",info['height'])
        for index in range(num_obj):
            for i in range(info['width']):
                for j in range(info['height']):
                    # print("image_id-->",image_id,"-i--->",i,"-j--->",j)
                    # print("info[width]----->",info['width'],"-info[height]--->",info['height'])
                    at_pixel = image.getpixel((i, j))
                    if at_pixel == index + 1:
                        mask[j, i, index] = 1
        return mask


    def load_shapes(self, count, img_floder, mask_floder, imglist, dataset_root_path):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # 修正済# Add classes
        #self.add_class("shapes", 1, "scrach")  #

        for i in range(len(args.CLASS_NAME)):
            self.add_class("shapes", i+1, args.CLASS_NAME[i])

        for i in range(count):

            filestr = imglist[i].split(".")[0]
            # print(imglist[i],"-->",cv_img.shape[1],"--->",cv_img.shape[0])
            # print("id-->", i, " imglist[", i, "]-->", imglist[i],"filestr-->",filestr)
            # filestr = filestr.split("_")[1]
            mask_path = mask_floder + "/" + filestr + ".png"
            yaml_path = dataset_root_path + "/labelme_json/" + filestr + "_json/info.yaml"
            print(dataset_root_path + "/labelme_json/" + filestr + "_json/img.png")
            cv_img = cv2.imread(dataset_root_path + "/labelme_json/" + filestr + "_json/img.png")

            self.add_image("shapes", image_id=i, path=img_floder + "/" + imglist[i],
                           width=cv_img.shape[1], height=cv_img.shape[0], mask_path=mask_path, yaml_path=yaml_path)


    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        #print("image_id", image_id)
        info = self.image_info[image_id]
        count = 1  # number of object
        img = Image.open(info['mask_path'])
        num_obj = self.get_obj_index(img)
        mask = np.zeros([info['height'], info['width'], num_obj], dtype=np.uint8)
        mask = self.draw_mask(num_obj, mask, img, image_id)
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count - 2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion

            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
        labels = []
        labels = self.from_yaml_get_class(image_id)
        labels_form = []
        
        #修正済み#
        #for i in range(len(labels)):
            #if labels[i].find("scrach") != -1:
                # print "box"
                #labels_form.append("scrach")

        for num in range(len(args.CLASS_NAME)):
            for i in range(len(labels)):
                if labels[i].find(args.CLASS_NAME[num]) != -1:
                    # print "box"
                    labels_form.append(args.CLASS_NAME[num])

        class_ids = np.array([self.class_names.index(s) for s in labels_form])
        return mask, class_ids.astype(np.int32)


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax

def train_model(init_weight=args.init_weight, epoch=args.epoch):
    dataset_root_path = os.path.join(ROOT_DIR,"train_data")
    img_floder = os.path.join(dataset_root_path, "pic")
    mask_floder = os.path.join(dataset_root_path, "cv2_mask")
    # yaml_floder = dataset_root_path
    imglist = os.listdir(img_floder)
    count = len(imglist)

    dataset_train = DrugDataset()
    dataset_train.load_shapes(count, img_floder, mask_floder, imglist, dataset_root_path)
    dataset_train.prepare()

    # print("dataset_train-->",dataset_train._image_ids)

    dataset_val = DrugDataset()
    dataset_val.load_shapes(round(count/4), img_floder, mask_floder, imglist, dataset_root_path)
    dataset_val.prepare()   

    # Create models in training mode
    config = ShapesConfig()
    config.display()
    model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)

    init_with = init_weight  # imagenet, coco, or last

    if init_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with == "coco":
        # Load weights trained on MS COCO, but skip layers that
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                    "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == "last":
        # Load the last models you trained and continue training
        checkpoint_file = model.find_last()
        model.load_weights(checkpoint_file, by_name=True)

    # Train the head branches
    # Passing layers="heads" freezes all layers except the head
    # layers. You can also pass a regular expression to select
    # which layers to train by name pattern.
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=epoch,
                layers='heads')

    # Fine tune all layers
    # Passing layers="all" trains all layers. You can also
    # pass a regular expression to select which layers to
    # train by name pattern.
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=epoch,
                layers="all")

if __name__ == '__main__':
    train_model()






