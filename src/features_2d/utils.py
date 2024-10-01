import cv2
from typing import Tuple, List
import numpy as np


def detectAndComputeKPandDescriptors(img: cv2.Mat, top_limit: int=0, bottom_limit: int=0) -> Tuple[List[cv2.KeyPoint], cv2.Mat]:
    """
    Detects keypoints and computes descriptors using AKAZE within a specified y-range.

    Args:
        img (cv2.Mat): Input image.
        top_limit (int): Top y-coordinate limit (inclusive).
        bottom_limit (int): Bottom y-coordinate limit (exclusive).

    Returns:
        Tuple[List[cv2.KeyPoint], cv2.Mat]: Detected keypoints and descriptors within the specified y-range.
    """
    if bottom_limit < 1: bottom_limit = img.shape[0]
    
    # Ensure the limits are within the image boundaries
    height = img.shape[0]
    top_limit = max(0, top_limit)
    bottom_limit = min(height, bottom_limit)
    if top_limit >= bottom_limit:
        raise ValueError("Invalid top and bottom limits: top_limit must be less than bottom_limit.")

    # Create a mask with the same dimensions as the input image
    mask = np.zeros(img.shape[:2], dtype=np.uint8)  # Shape: (height, width)

    # Set the mask to 255 (white) in the y-range between top_limit and bottom_limit
    mask[top_limit:bottom_limit, :] = 255

    # Initialize the AKAZE feature detector
    akaze = cv2.AKAZE_create()

    # Detect keypoints and compute descriptors within the mask
    kpts, desc = akaze.detectAndCompute(img, mask)

    return kpts, desc



def detectAndComputeKPandDescriptors_new(img: cv2.Mat, top_limit: int = 0, bottom_limit: int = 0,wished_nb_kpts = 250,nb_zones = 4) -> Tuple[List[cv2.KeyPoint], cv2.Mat]:
    """
    Detects keypoints and computes descriptors using AKAZE, splitting the image horizontally into 4 zones.
    Retains the top 100 keypoints from each zone, and groups them together before returning.

    Args:
        img (cv2.Mat): Input image.
        top_limit (int): Top y-coordinate limit (inclusive).
        bottom_limit (int): Bottom y-coordinate limit (exclusive).

    Returns:
        Tuple[List[cv2.KeyPoint], cv2.Mat]: Detected keypoints and descriptors from all zones.
    """

    nb_zones = 4
    nb_kpts_per_zone = int(wished_nb_kpts/nb_zones)

    if bottom_limit < 1:
        bottom_limit = img.shape[0]

    # Ensure the limits are within the image boundaries
    height,width = img.shape[:2]
    top_limit = max(0, top_limit)
    bottom_limit = min(height, bottom_limit)

    if top_limit >= bottom_limit:
        raise ValueError("Invalid top and bottom limits: top_limit must be less than bottom_limit.")

    # Create a mask with the same dimensions as the input image
    mask = np.zeros(img.shape[:2], dtype=np.uint8)  # Shape: (height, width)

    # Set the mask to 255 (white) in the y-range between top_limit and bottom_limit
    mask[top_limit:bottom_limit, :] = 255

    # Initialize the AKAZE feature detector
    akaze = cv2.AKAZE_create()

    # Split the image into 4 horizontal zones
    zone_width = width // nb_zones
    all_kpts = []
    all_desc = []

    for i in range(nb_zones):
        zone_left = i * zone_width
        zone_right = zone_left + zone_width

        # Ensure the last zone extends to the exact bottom_limit
        if i == 3:
            zone_right = width

        # Create a mask for the current zone
        zone_mask = np.zeros_like(mask)
        zone_mask[:,zone_left:zone_right] = 255

        # Detect keypoints and compute descriptors within the zone
        kpts, desc = akaze.detectAndCompute(img, zone_mask)

        if kpts and desc is not None:
            # Retain only the top 100 keypoints based on their response (best ones)
            if len(kpts) > nb_kpts_per_zone:
                kpts, desc = zip(*sorted(zip(kpts, desc), key=lambda x: x[0].response, reverse=True)[:nb_kpts_per_zone])
                desc = np.array(desc)  # Convert descriptors back to a NumPy array

            # Append the keypoints and descriptors to the final list
            all_kpts.extend(kpts)
            if len(all_desc) == 0:
                all_desc = desc
            else:
                all_desc = np.vstack((all_desc, desc))

    return all_kpts, all_desc if len(all_desc) > 0 else None

def getMatches(desc1: cv2.Mat, desc2: cv2.Mat, nn_match_ratio=0.5) -> List[cv2.DMatch]:
    """
    Finds matches between descriptors using brute-force hamming distance.

    Args:
    - desc1: Descriptors from the first image (cv2.Mat).
    - desc2: Descriptors from the second image (cv2.Mat).
    - nn_match_ratio: Nearest neighbor match ratio threshold.

    Returns:
    - good_matches: List of good matches (cv2.DMatch).
    """
    # Create the matcher with brute-force Hamming distance
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)

    # Find the two nearest neighbors for each descriptor
    nn_matches = matcher.knnMatch(desc1, desc2, 2)

    # Use list comprehension to filter good matches
    good_matches = [m for m, n in nn_matches if m.distance < nn_match_ratio * n.distance]

    return good_matches



