import cv2 as cv
from typing import Tuple, List

def detectAndCompute(img: cv.Mat) -> Tuple[List[cv.KeyPoint], cv.Mat]:
    """
    Detects keypoints and computes descriptors using AKAZE.

    Args:
        img (cv.Mat): Input image.

    Returns:
        Tuple[List[cv.KeyPoint], cv.Mat]: Detected keypoints and descriptors.
    """
    akaze = cv.AKAZE_create()
    kpts, desc = akaze.detectAndCompute(img, None)
    return kpts, desc

def getMatches(desc1: cv.Mat, desc2: cv.Mat) -> List[List[cv.DMatch]]:
    """
    Finds matches between descriptors using brute-force hamming distance.

    Args:
        desc1 (cv.Mat): Descriptors from the first image.
        desc2 (cv.Mat): Descriptors from the second image.

    Returns:
        List[List[cv.DMatch]]: List of matches.
    """
    matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_BRUTEFORCE_HAMMING)
    nn_matches = matcher.knnMatch(desc1, desc2, 2)
    return nn_matches
