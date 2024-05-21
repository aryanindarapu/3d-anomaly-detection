import cv2
import numpy as np

img = cv2.imread('data/mvtec_3d_anomaly_detection/bagel/test/combined/gt/000.png')
rgb_img = cv2.imread('data/mvtec_3d_anomaly_detection/bagel/test/combined/rgb/000.png')
# print(img.shape)
print(np.unique(img))