import cv2
from typing import Tuple, List
import numpy as np

from src.road_detection.common import AttentionWindow

class DescriptorManager:
    feature2D:cv2.Feature2D
    kpts:List[cv2.KeyPoint]=[]
    desc:List[cv2.Mat]=[]
    matches: List[cv2.DMatch]=[]
    def detectAndComputeKPandDescriptors(self,img: cv2.Mat, attentionWindow:AttentionWindow) -> Tuple[List[cv2.KeyPoint], cv2.Mat]:
        """
        Detects keypoints and computes descriptors within a specified y-range.

        Args:
            img (cv2.Mat): Input image.
            attentionWindow (AttentionWindow): Attention window object containing the window to considerate.

        Returns:
            Tuple[List[cv2.KeyPoint], cv2.Mat]: Detected keypoints and descriptors within the specified y-range.
        """
        mask = None
        if attentionWindow is not None:
            # Create a mask with the same dimensions as the input image
            mask = np.zeros(img.shape[:2], dtype=np.uint8)  # Shape: (height, width)

            # Set the mask to 255 (white) in the y-range between top_limit and bottom_limit
            top = attentionWindow.top
            bottom = attentionWindow.bottom
            left = attentionWindow.left
            right = attentionWindow.right
            mask[top:bottom, left:right] = 255

        # Initialize the AKAZE feature detector


        # Detect keypoints and compute descriptors within the mask
        self.kpts, self.desc = self.feature2D.detectAndCompute(img, mask)
        return self.kpts, self.desc
    
    def detectAndComputeKPandDescriptors_zone(self,img: cv2.Mat, top_limit: int = 0, bottom_limit: int = 0,wished_nb_kpts = 1000,nb_zones = 4) -> Tuple[List[cv2.KeyPoint], cv2.Mat]:
        
        """
        Detects keypoints and computes descriptors , splitting the image horizontally into 4 zones.
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

        height,width = img.shape[:2]

        if bottom_limit < 1:
            bottom_limit = height

        # Ensure the limits are within the image boundaries
        top_limit = max(0, top_limit)
        bottom_limit = min(height, bottom_limit)

        if top_limit >= bottom_limit:
            raise ValueError("Invalid top and bottom limits: top_limit must be less than bottom_limit.")

        # Create a mask with the same dimensions as the input image
        mask = np.zeros(img.shape[:2], dtype=np.uint8)  # Shape: (height, width)

        # Set the mask to 255 (white) in the y-range between top_limit and bottom_limit
        mask[top_limit:bottom_limit, :] = 255

        # Split the image into 4 horizontal zones
        zone_width = width // nb_zones
        self.kpts = []
        self.desc = []

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
            kpts, desc = self.feature2D.detectAndCompute(img, zone_mask)

            if kpts and desc is not None:
                # Retain only the top 100 keypoints based on their response (best ones)
                if len(kpts) > nb_kpts_per_zone:
                    kpts, desc = zip(*sorted(zip(kpts, desc), key=lambda x: x[0].response, reverse=True)[:nb_kpts_per_zone])
                    desc = np.array(desc)  # Convert descriptors back to a NumPy array

                # Append the keypoints and descriptors to the final list
                self.kpts.extend(kpts)
                if len(self.desc) == 0:
                    self.desc = desc
                else:
                    self.desc = np.vstack((self.desc, desc))

        return self.kpts, self.desc if len(self.desc) > 0 else None
    
    def getMatches(self,desc1: cv2.Mat, desc2: cv2.Mat) -> List[cv2.DMatch]:
        pass

class AkazeDescriptorManager(DescriptorManager):

    def __init__(self):
        super().__init__()
        self.feature2D = cv2.AKAZE_create()

    def getMatches(self,desc1: cv2.Mat, desc2: cv2.Mat, nn_match_ratio=0.75) -> List[cv2.DMatch]:
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
        self.matches= [m for m, n in nn_matches if m.distance < nn_match_ratio * n.distance]

        return self.matches

class OrbDescriptorManager(DescriptorManager):
    def __init__(self,nfeatures=5000):
        super().__init__()
        self.feature2D = cv2.ORB_create(nfeatures = nfeatures)

    def detectAndComputeKPandDescriptors(self,img: cv2.Mat, attentionWindow:AttentionWindow) -> Tuple[List[cv2.KeyPoint], cv2.Mat]:
        nfeatures = int(attentionWindow.area/200)
        self.feature2D = cv2.ORB_create(nfeatures = nfeatures)
        return super().detectAndComputeKPandDescriptors(img, attentionWindow)

    def getMatches(self, desc1: cv2.Mat, desc2: cv2.Mat, nn_match_ratio=0.75) -> List[cv2.DMatch]:
        """
        Finds matches between descriptors using brute-force Hamming distance.

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

        # Filter good matches using nearest neighbor ratio
        self.matches = [m for m, n in nn_matches if m.distance < nn_match_ratio * n.distance]

        return self.matches


def detectAndComputeKPandDescriptors(img: cv2.Mat, top_limit: int=0, bottom_limit: int=0) -> Tuple[List[cv2.KeyPoint], cv2.Mat]:
   akazeDescriptorManager  = AkazeDescriptorManager()
   return akazeDescriptorManager.detectAndComputeKPandDescriptors(img, top_limit, bottom_limit)

def detectAndComputeKPandDescriptors_new(img: cv2.Mat, top_limit: int = 0, bottom_limit: int = 0,wished_nb_kpts = 1000,nb_zones = 4) -> Tuple[List[cv2.KeyPoint], cv2.Mat]:
   akazeDescriptorManager  = AkazeDescriptorManager()
   return akazeDescriptorManager.detectAndComputeKPandDescriptors_zone(img, top_limit, bottom_limit,wished_nb_kpts=wished_nb_kpts,nb_zones=nb_zones)

def getMatches(desc1: cv2.Mat, desc2: cv2.Mat, nn_match_ratio=0.5) -> List[cv2.DMatch]:
    akazeDescriptorManager  = AkazeDescriptorManager()
    return akazeDescriptorManager.getMatches(desc1, desc2, nn_match_ratio=nn_match_ratio)