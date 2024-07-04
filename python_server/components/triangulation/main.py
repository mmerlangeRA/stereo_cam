import csv
import os
from typing import List, Tuple
import numpy as np
import cv2 as cv
from scipy.optimize import minimize
from src.utils.path_utils import find_image_path
from src.triangulate.features import detectAndCompute, getMatches
from src.triangulate.calibrate import calibrate_left_right, getCalibrationFrom3Matching, load_calibration_params, save_calibration_params
from src.triangulate.triangulate import rotation_matrix_from_params, triangulate_from_rays,get_3d_point_cam1_2_from_coordinates
from pydantic import BaseModel, Field


class TriangulationRequest(BaseModel):
    keypoints_cam1: Tuple[float, float] = Field(..., example=(0, 0))
    keypoints_cam2: Tuple[float, float] = Field(..., example=(0, 0))
    image_width: int = Field(..., example=1280)
    image_height: int = Field(..., example=1280)
    R: List[float] = Field(..., example=[0, 0, 0])
    t: List[float] = Field(..., example=[0, 0, 0])

    class Config:
        from_attributes = True

class AutoCalibrationRequest(BaseModel):
    imgLeft_name: str = Field(..., example="D_P5_CAM_G_2_EAC.png")
    imgRight_name: str = Field(..., example="D_P5_CAM_D_2_EAC.png")
    initial_params: List[float] = Field(..., example=[0, 0, 0, 1.12,0,0])
    bnds: List[tuple] = Field(..., example=[(- np.pi*10./180.,  np.pi*10./180.), (- np.pi*10./180.,  np.pi*10./180.),(- np.pi*10./180.,  np.pi*10./180.),(1.11, 1.13),(- 0.12001, 0.12001),(- 0.12001, 0.12001)])
    inlier_threshold:float = Field(..., example=0.01)

    class Config:
        from_attributes = True

def auto_calibrate(request:AutoCalibrationRequest)->List[float]:
    photo_folder = os.path.join(os.getcwd(), 'Photos')
    img1_path = find_image_path(photo_folder, request.imgLeft_name)
    img2_path = find_image_path(photo_folder, request.imgRight_name)
    if img1_path is None or img2_path is None:
        raise ValueError("Image not found")
    left_image = cv.imread(img1_path)
    right_image = cv.imread(img2_path)
    best_results = calibrate_left_right(left_image, right_image, request.initial_params, request.bnds,request.inlier_threshold)
    optimized_params = best_results["params"]
    if isinstance(optimized_params, np.ndarray):
        optimized_params = optimized_params.tolist()
    
    return optimized_params


def triangulatePoints(request:TriangulationRequest)-> Tuple[List[float], List[float], float]:
    rot_matrix = rotation_matrix_from_params(request.R)
    point1,point2,residual = get_3d_point_cam1_2_from_coordinates(request.keypoints_cam1, request.keypoints_cam2, request.image_width, request.image_height, rot_matrix, request.t, True)
    if type(point1) is np.ndarray:
        point1 = point1.tolist()
    if type(point2) is np.ndarray:
        point2 = point2.tolist()
    return point1, point2, residual