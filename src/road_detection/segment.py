import cv2
from matplotlib import pyplot as plt
import numpy as np
from src.pidnet.main import segment_image
from src.utils.path_utils import get_static_folder_path
from typing import Tuple, List, Optional
import numpy.typing as npt
from src.road_detection.seg_former import seg_segment_image
from src.road_detection.common import AttentionWindow

def segment_road_image(img: npt.NDArray[np.uint8],kernel_width=10,use_seg=True, debug=False) -> npt.NDArray[np.uint8]:
    """
    Computes binary image corresponding to road.

    Parameters:
    - img: rgb image
    returns
    - binary image

    """

    if use_seg:
        segmented_image, pred= seg_segment_image(img,verbose=debug)
    else:
        segmented_image, pred = segment_image(img)

    if debug:
        print(f'segment_road_image using segFormer={use_seg}')
        img_name = "segmented_seg.png" if use_seg else "segmented_no_seg.png"
        cv2.imwrite(get_static_folder_path(img_name), segmented_image)
        cv2.imshow("segmented_image",segmented_image)
    # Create a mask for the road class
    road_mask = (pred == 0).astype(np.uint8)
    # Check if the mask has the same dimensions as the segmented image
    assert road_mask.shape == segmented_image.shape[:2], "Mask size does not match the image size."

    masked_segmented = cv2.bitwise_and(segmented_image, segmented_image, mask=road_mask)

    gray = cv2.cvtColor(masked_segmented, cv2.COLOR_BGR2GRAY)
    # we clean the image by removing small blobs

    kernel_size = (kernel_width,kernel_width) #this size could be computed dynamically from image size

    ret, thresh = cv2.threshold(gray, 1, 255, 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    # Apply the dilation operation to the edged image
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    return thresh


def test_segmentation(img_path:str=r'C:\Users\mmerl\projects\stereo_cam\Photos\P5\D_P5_CAM_G_0_EAC.png' ) -> None:
    img = cv2.imread(get_static_folder_path(img_path))

    height, width = img.shape[:2]
    max_width = 2048
    max_height = int(height/width*max_width)
    img = cv2.resize(img, (max_width, max_height))
    height, width = img.shape[:2]

    window = AttentionWindow(int(0.4*width), int(0.6*width), int(0.3*height), int(0.6*height))
    windowed = window.crop_image(img)
    thresh_windowed=segment_road_image(windowed, debug=True)
    thresh = np.zeros(img.shape[:2], dtype=np.uint8)
    thresh[window.top:window.bottom, window.left:window.right] = thresh_windowed
    cv2.imshow("thresh", thresh)
    cv2.waitKey(0)
    