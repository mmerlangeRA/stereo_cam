import cv2
from typing import Tuple, List

def detectAndComputeKPandDescriptors(img: cv2.Mat) -> Tuple[List[cv2.KeyPoint], cv2.Mat]:
    """
    Detects keypoints and computes descriptors using AKAZE.

    Args:
        img (cv2.Mat): Input image.

    Returns:
        Tuple[List[cv2.KeyPoint], cv2.Mat]: Detected keypoints and descriptors.
    """
    akaze = cv2.AKAZE_create()
    kpts, desc = akaze.detectAndCompute(img, None)
    return kpts, desc

def getMatches(desc1: cv2.Mat, desc2: cv2.Mat,nn_match_ratio=0.5) -> List[List[cv2.DMatch]]:
    """
    Finds matches between descriptors using brute-force hamming distance.

    Args:
        desc1 (cv2.Mat): Descriptors from the first image.
        desc2 (cv2.Mat): Descriptors from the second image.

    Returns:
        List[List[cv2.DMatch]]: List of matches.
    """
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
    nn_matches = matcher.knnMatch(desc1, desc2, 2)
    good_matches=[]
    for i, (m, n) in enumerate(nn_matches):
        if m.distance < nn_match_ratio * n.distance:
            good_matches.append(m)
    return good_matches


