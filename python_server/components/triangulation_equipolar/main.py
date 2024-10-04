from typing import List, Tuple
import numpy as np
import cv2
from python_server.utils.path_helper import get_uploaded_photos_path
from src.triangulate.main import get_3d_point_cam1_2_from_coordinates
from pydantic import BaseModel, Field
from python_server.settings.settings import settings
from src.calibration.equirectangular.main import auto_compute_cam2_transform
from src.utils.coordinate_transforms import rotation_matrix_from_vector3D

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

def auto_calibrate_equipoloar(request:AutoCalibrationRequest, verbose=False)->List[float]:
    imgLeft_path = request.imgLeft_name if "/" in request.imgLeft_name else get_uploaded_photos_path(request.imgLeft_name)
    imgRight_path = request.imgRight_name if "/" in request.imgRight_name else get_uploaded_photos_path(request.imgRight_name)

    try:
        left_image = cv2.imread(imgLeft_path)
    except Exception as e:
        raise FileNotFoundError(f"left_image not found {e}")
    try:
        right_image = cv2.imread(imgRight_path)
    except Exception as e:
        raise FileNotFoundError(f"right_image not found{e}")

    best_results = auto_compute_cam2_transform(left_image, right_image, request.initial_params, request.bnds,request.inlier_threshold,verbose=verbose)
    optimized_params = best_results["params"]
    if isinstance(optimized_params, np.ndarray):
        optimized_params = optimized_params.tolist()
    
    return optimized_params

def triangulate_equipolar_points(request:TriangulationRequest, verbose=False)-> Tuple[List[float], List[float], float]:
    rot_matrix = rotation_matrix_from_vector3D(request.R)
    point1,point2,residual_in_m = get_3d_point_cam1_2_from_coordinates(request.keypoints_cam1, request.keypoints_cam2, request.image_width, request.image_height, rot_matrix, request.t, verbose)
    if type(point1) is np.ndarray:
        point1 = point1.tolist()
    if type(point2) is np.ndarray:
        point2 = point2.tolist()
    return point1, point2, residual_in_m