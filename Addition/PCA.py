# The implementation of Principle Component
# Analysis 
# Author : DMAN

import numpy as np
import cv2 as cv


image = cv.imread("test.png")
print(image.shape)


class PCA():
    def __init__(self, data, k):
        self.matrix = np.dot(np.transpose(data), data)
        self.image = data
        self.k = k

    def pca(self):
        pass