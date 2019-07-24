import os
import sys

# To find local version
ROOT_DIR = os.path.abspath("../")
MaskRCNN_DIR = os.path.abspath("../Mask_RCNN")
sys.path.append(os.path.join(MaskRCNN_DIR, "samples/coco/"))

sys.path.append(MaskRCNN_DIR)  # To find local version of the library
MODEL_DIR = os.path.join(MaskRCNN_DIR, "samples/coco/")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

from mrcnn import visualize
import mrcnn.model as modellib
from mrcnn import utils
# from samples.coco import coco as coco
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

import itertools
import colorsys

from skimage.measure import find_contours
from skimage import measure
from matplotlib import patches,  lines
from matplotlib.patches import Polygon
# from mrcnn import model
from PIL import Image

import coco

import tensorflow as tf
import cv2

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


# config = InferenceConfig()

# model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
# model.load_weights(COCO_MODEL_PATH, by_name=True)


class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


def random_colors(N):
    hsv = [(i / N, 1, 1) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


def apply_mask_image(bg, image, mask,):
    """Apply the given mask to the image.
    """
    for c in range(3):
        bg[:, :, c] = np.where(mask == 1,
                               image[:, :, c], bg[:, :, c],)
    return bg


def apply_mask_inverse_image(bg, image, mask,):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  bg[:, :, c], image[:, :, c],)
    return image


def load_object(file_name, model):
    # model = modellib.MaskRCNN(
    #     mode="inference", model_dir=MODEL_DIR, config=config)
    image = load_img(file_name)
    print(image)
    # model = os.path.dirname
    results = model.detect([image], verbose=1)
    r = results[0]
    # visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
    #                             class_names, r['scores'])
    print(r)
    N = len(r['rois'])
    colors = random_colors(N)
    figsize = (16, 16)
    _, ax = plt.subplots(1, figsize=figsize)

    height, width = image.shape[:2]
    ax.axis('off')
    ax.margins(0, 0)
    color = (.2, 0.5, 0.9)
    captions = None
    masked_image = image.astype(np.uint32).copy()
    counts = {}
    output = []
    for i in range(N):
        y1, x1, y2, x2 = r['rois'][i]
        if not captions:
            caption = class_names[r['class_ids'][i]]
            if caption not in counts:
                counts[caption] = 1
                caption = caption+str(counts[caption])
            else:
                counts[caption] += 1
                caption = caption+str(counts[caption])
        else:
            caption = captions[i]
        ax.text(x1, y1 + 8, caption,
                color='w', size=11, backgroundcolor="none")
        output.append(caption)

        mask = r['masks'][:, :, i]
        masked_image = apply_mask(masked_image, mask, colors[i])

        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)

    fig = ax.imshow(masked_image.astype(np.uint8))
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.savefig('static/out/all_objects.jpg', bbox_inches='tight',
                pad_inches=0)
    all = 'static/out/all_objects.jpg'
    return r, all


# Contour Outline
def show_selection_outlines(raw_input, image, r):
    image = skimage.io.imread(image)
    figsize = (16, 16)
    _, ax = plt.subplots(1, figsize=figsize)

    # color = (.2, 0.5, 0.9)
    masked_image = image.astype(np.uint32).copy()
    contour_outlines = []
    if raw_input == 1000:
        raw_input = range(len(r['rois']))
    for i in raw_input:
        if i > len(r['rois']) or i < 0:
            continue
        y1, x1, y2, x2 = r['rois'][i]
        mask = r['masks'][:, :, i]
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        contour_outlines.append(contours)
        for n, contour in enumerate(contours):
            ax.plot(contour[:, 1], contour[:, 0], linewidth=2,)

    fig = ax.imshow(masked_image.astype(np.uint8))
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    # ax.axes.get_xaxis().set_visible(False)
    # ax.axes.get_yaxis().set_visible(False)
    plt.savefig('static/out/selected_objects.jpg', bbox_inches='tight',
                pad_inches=0)
    outlines = 'static/out/selected_objects.jpg'
    return outlines


# Crop image according to selected contours
def show_selection_crop(raw_input, image, r):
    image = skimage.io.imread(image)
    figsize = (16, 16)
    _, ax = plt.subplots(1, figsize=figsize)
    ax.axis('off')
    ax.margins(0, 0)
    color = (.2, 0.5, 0.9)
    contour_outlines = []
    background_image = np.zeros_like(image)
    # background_image[:, :, ] = [0, 0, 0]
    masked_image = image.astype(np.uint32).copy()
    contour_outlines = []
    if raw_input == 1000:
        raw_input = range(len(r['rois']))
    for i in raw_input:
        if i > len(r['rois']) or i < 0:
            continue
        y1, x1, y2, x2 = r['rois'][i]
        mask = r['masks'][:, :, i]
        background_image = apply_mask_image(
            background_image, masked_image, mask,)

    fig = ax.imshow(background_image.astype(np.uint8))
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.savefig('static/out/crop.jpg', bbox_inches='tight', pad_inches=0)
    location = 'static/out/crop.jpg'
    return location, background_image

# Crop image according to selected inverse contours


def show_selection_inverse(raw_input, image, r):
    image = skimage.io.imread(image)
    figsize = (16, 16)
    _, ax = plt.subplots(1, figsize=figsize)

    ax.axis('off')
    ax.margins(0, 0)
    color = (.2, 0.5, 0.9)
    contour_outlines = []
    background_image = np.zeros_like(image)
    # background_image[:, :, ] = [0, 0, 0]
    masked_image = image.astype(np.uint32).copy()
    contour_outlines = []
    if raw_input == 1000:
        raw_input = range(len(r['rois']))
    for i in raw_input:
        if i > len(r['rois']) or i < 0:
            continue
        y1, x1, y2, x2 = r['rois'][i]
        mask = r['masks'][:, :, i]
        masked_image = apply_mask_inverse_image(
            background_image, masked_image, mask,)

    fig = ax.imshow(masked_image.astype(np.uint8))
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.savefig('static/out/crop_inverse.jpg',
                bbox_inches='tight', pad_inches=0)
    location = 'static/out/crop_inverse.jpg'
    return location, masked_image


def load_img(path_to_img):
    img = skimage.io.imread(path_to_img)
    return img


def blending(crop_path, original_path, style_path):
    print('~~~~~~~~ \n\n crop_path original_path style_path = ', crop_path)
    print(original_path)
    print(style_path)

    # styled = cv2.imread(style_path)
    # crop = cv2.imread(crop_path)
    # original = cv2.imread(original_path)    
    
    crop = cv2.imread(os.path.join(os.path.abspath(""), crop_path)).astype('uint8')
    original = cv2.imread(os.path.join(os.path.abspath(""), original_path)).astype('uint8')
    styled = cv2.imread(os.path.join(os.path.abspath(""),style_path)).astype('uint8')
    # styled = cv2.resize(styled, (original.shape[1], original.shape[0]))

    print(styled.shape)
    print(original.shape)
    print(styled, original, crop)
    crop = cv2.resize(crop, (styled.shape[1], styled.shape[0]))
    original = cv2.resize(original, (styled.shape[1], styled.shape[0]))

    non_black_pixels_mask = np.any(np.logical_and(
        crop != [0, 0, 0], crop != [255, 255, 255]),  axis=-1)

    original_copy = original
    mask = crop
    styled = styled.astype(float)
    # styled = styled.reshape(styled.shape[0], styled.shape[1], 3)
    # original_copy = original_copy.reshape(
    #     original_copy.shape[1], original_copy.shape[2], 3)
    original_copy = original_copy.astype(float)

    mask[non_black_pixels_mask] = [255, 255, 255]
    m = mask
    # m = m.reshape(original.shape[1], original.shape[2], 3)

    # cv2.imwrite('static/out/bin-mask-str.jpg', m)
    blurSigma = 5
    m = m.astype(float)/255.0
    # m = cv2.imread('static/out/bin-mask-str.jpg').astype(float)/255.0
    m = cv2.GaussianBlur(m, (2*blurSigma+1, 2*blurSigma+1), blurSigma)

    # apply alpha blending
    style_layer = cv2.multiply(m, styled)
    regular_layer = cv2.multiply(1.0-m, original_copy)
    out = style_layer + regular_layer

    out = out.astype('uint8')
    output_str = 'static/final/styled_final-1.jpg'
    # out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_str, out)
    return output_str


# show_objects = load_object(filename, model)
# contour_outlines = show_selection(raw_input, filename, show_objects)
