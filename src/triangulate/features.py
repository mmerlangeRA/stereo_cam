import os
import numpy as np
import cv2 as cv

def detectAndCompute(img:cv.Mat):
    akaze = cv.AKAZE_create()
    kpts, desc = akaze.detectAndCompute(img, None)
    return kpts, desc

def getMatches(desc1, desc2):
    matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_BRUTEFORCE_HAMMING)
    nn_matches = matcher.knnMatch(desc1, desc2, 2)
    return nn_matches


