import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

dir = "./dataset/training_set"

categories = ['cats', 'dogs']

for category in categories:
    path = os.path.join(dir, category)

    for img in os.listdir(path):
        imgpath = os.path.join(path, img)
        petimg = cv2.imread(imgpath, 0)
        cv2.imshow('image', petimg)
        break
    break

cv2.waitKey(0)
cv2.destroyAllWindows()