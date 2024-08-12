import cv2
#from src.road_detection.segment import test_segmentation
from src.calibration.stereo_standard_refinement import compute_auto_calibration_for_2_stereo_standard_images



imgL = cv2.imread(r"C:\Users\mmerl\projects\stereo_cam\undistorted_CUBE\11_rectified_left.jpg",cv2.IMREAD_COLOR)

imgR =cv2.imread(r"C:\Users\mmerl\projects\stereo_cam\undistorted_CUBE\11_rectified_right.jpg",cv2.IMREAD_COLOR)
height,width = imgL.shape[:2]

K,cost,refined_rvec,refined_tvec=compute_auto_calibration_for_2_stereo_standard_images(imgLeft=imgL, imgRight=imgR)
new_width = 600
new_height = int((height / width) * new_width)

ratio = new_width/width

print("shape",imgL.shape)
print("ratio",ratio)

imgL = cv2.resize(imgL, (new_width, new_height))
imgR = cv2.resize(imgR, (new_width, new_height))
print("new shape",imgL.shape)
""" imgL = imgL[:480, :]
imgR = imgR[:480, :] """
# cv2.imshow('imgL', imgL)
# cv2.imshow('imgR', imgR)
# cv2.waitKey(0)
K,cost,refined_rvec,refined_tvec=compute_auto_calibration_for_2_stereo_standard_images(imgLeft=imgL, imgRight=imgR)