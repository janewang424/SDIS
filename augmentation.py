# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 01:57:31 2020

@author: janew
"""

import cv2
import numpy as np


def do_identity(image):
    return image


def do_horizon_flip(image, p=0.5):
    if np.random.uniform(0, 1) < p:
        image = cv2.flip(image, 1, dst=None)
        return image
    else:
        return image


def do_vertical_flip(image, p=0.5):
    if np.random.uniform(0, 1) < p:
        image = cv2.flip(image, 0, dst=None)
        return image
    else:
        return image


def do_diagonal_flip(image, p=0.5):
    if np.random.uniform(0, 1) < p:
        image = cv2.flip(image, -1, dst=None)
        return image
    else:
        return image


def do_random_rotate(image, magnitude=0.5, p=0.5):
    if np.random.uniform(0, 1) < p:
        angle = np.random.uniform(-1, 1)*180*magnitude

        height, width = image.shape[:2]
        cx, cy = width // 2, height // 2

        transform = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        image = cv2.warpAffine(image, transform, (width, height), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return image


def do_CLAHE(image, clipLimit=2.0, tileGridSize=(8,8), p=0.5):
    if np.random.uniform(0, 1) < p:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

        gryimg = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        gryimg_planes = cv2.split(gryimg)
        gryimg_planes[0] = clahe.apply(gryimg_planes[0])
        gryimg = cv2.merge(gryimg_planes)
        image = cv2.cvtColor(gryimg, cv2.COLOR_LAB2BGR)
        return image # Random CLAHE Contrast Limited Adaptive Histogram Equalization
    else:
        return image
