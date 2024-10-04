from abc import abstractmethod
import time
import cv2
import numpy as np
import torch
from src.pidnet.main import segment_image
from src.utils.path_utils import get_ouput_path, get_static_folder_path
from typing import Tuple, List, Optional
import numpy.typing as npt
from src.road_detection.seg_former import SegFormerEvaluater

class RoadSegmentator:
    kernel_width: int
    debug: bool
    def __init__(self, kernel_width=10, debug=False):
        self.kernel_width = kernel_width
        self.debug = debug
    
    @abstractmethod
    def get_segmented_image_and_pred_and_road_label(self, img: npt.NDArray[np.uint8]) -> Tuple[npt.NDArray[np.uint8], any, int]:
        pass

    
    def segment_road_image(self, img: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        """
            Computes binary image corresponding to road.
            Parameters:
            - img: rgb image
            returns
            - binary image
        """
        segmented_image, pred,road_label = self.get_segmented_image_and_pred_and_road_label(img)
        if self.debug:
            img_name = "segmented.png" 
            cv2.imwrite(get_ouput_path(img_name), segmented_image)
            #cv2.imshow("segmented_image",segmented_image)
        # Create a mask for the road class
        road_mask = (pred == road_label).astype(np.uint8)
        # Check if the mask has the same dimensions as the segmented image
        assert road_mask.shape == segmented_image.shape[:2], "Mask size does not match the image size."

        masked_segmented = cv2.bitwise_and(segmented_image, segmented_image, mask=road_mask)

        gray = cv2.cvtColor(masked_segmented, cv2.COLOR_BGR2GRAY)
        # we clean the image by removing small blobs

        kernel_size = (self.kernel_width,self.kernel_width) #this size could be computed dynamically from image size

        ret, thresh = cv2.threshold(gray, 1, 255, 0)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        # Apply the dilation operation to the edged image
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        return thresh

class PIDNetRoadSegmentator(RoadSegmentator):
    def __init__(self, kernel_width=10, debug=False):
        super().__init__(kernel_width, debug)
        print(f"Creating PIDNetRoadSegmentator")

    def get_segmented_image_and_pred_and_road_label(self, img: npt.NDArray[np.uint8]) -> Tuple[npt.NDArray[np.uint8], any, int]:
        segmented_image, pred,road_label = segment_image(img)
        return segmented_image, pred, road_label
    
class SegFormerRoadSegmentator(RoadSegmentator):
    use_1024 : bool
    segFormerEvaluater: SegFormerEvaluater
    def __init__(self, kernel_width=10, use_1024=False, debug=False):
        super().__init__(kernel_width, debug)
        self.use_1024 = use_1024
        if self.debug:
            print(f"Creating SegFormerRoadSegmentator with {use_1024}")
        self.segFormerEvaluater = SegFormerEvaluater(use_1024=use_1024)

    def get_segmented_image_and_pred_and_road_label(self, cv_image: npt.NDArray[np.uint8]) -> Tuple[npt.NDArray[np.uint8], any, int]:
        original_height, original_width = cv_image.shape[:2]
        logits_resized, road_label = self.segFormerEvaluater.segment_image(cv_image=cv_image, use_1024=self.use_1024)
        # Get the predicted labels
        pred_labels = torch.argmax(logits_resized, dim=1)  # shape (batch_size, height, width)
        pred_labels = pred_labels.squeeze().cpu().numpy()  # Convert to numpy array and remove batch dimension
        # Define a color map (cityscapes has 19 classes)
        colors = np.array([(128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153),
                        (153, 153, 153), (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152),
                        (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
                        (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32)])

        # Create an empty array to hold the segmented image
        segmented_image = np.zeros((original_height, original_width, 3), dtype=np.uint8)

        # Map each label to the corresponding color
        for label in range(len(colors)):
            segmented_image[pred_labels == label] = colors[label]
        return segmented_image, pred_labels, road_label

