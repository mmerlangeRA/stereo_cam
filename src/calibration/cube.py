import os
from typing import List
import numpy as np
import cv2
import yaml

from src.utils.cube_image import get_cube_front_image, get_cube_sub_images


calibration_folder = "calibration"

def save_calibration_params(params, filename:str)->None:
    np.savetxt(filename+'.csv', params, delimiter=',')

def load_calibration_params(filename:str)->np.ndarray[float]:
    loaded =np.loadtxt(filename+'.csv', delimiter=',')
    return loaded

def save_calibration(mtx, dist, rmse,fname="calibration_matrix.yaml")->None:
    data = {'camera_matrix': np.asarray(mtx).tolist(),
        'dist_coeff': np.asarray(dist).tolist(),'rmse':rmse}
    # and save it to a file
    with open(fname, "w") as f:
        yaml.dump(data, f)

def read_calibration(fname="calibration_matrix.yaml")->np.ndarray[float]:
    with open(fname, 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return np.asarray(data['camera_matrix']), np.asarray(data['dist_coeff']),data['rmse']


def compute_cube_calibration(image_paths:List[str],chessboard_size:cv2.typing.Size,square_size:float)->tuple[float,cv2.typing.MatLike, cv2.typing.MatLike]:
    print(len(image_paths))
    nb_used = 0
    objpoints=[]
    imgpoints=[]
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
    objp *= square_size 
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    for fname in image_paths:
        print(fname)
        img = cv2.imread(fname)
        
        front_image=get_cube_front_image(img)
        gray = cv2.cvtColor(front_image, cv2.COLOR_BGR2GRAY)
        th,tw=gray.shape[:2]
        print(f'th={th}, tw={tw}')
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            #ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
            print(f'nb corners={len(corners) if corners is not None else 0}')
            objpoints.append(objp.copy())
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
            # Draw and display the corners
            #cv2.drawChessboardCorners(gray, chessboard_size, corners2, ret)
            #cv2.imshow('img', gray)
            #cv2.waitKey(500)
            nb_used+=1


    print(f'nb used={nb_used} out of {len(image_paths)}')
    print(f'nb objpoints={len(objpoints)}, nb imgpoints {len(imgpoints)}')
    # Check if we have enough points for calibration
    if len(objpoints) > 0 and len(imgpoints) > 0 and len(objpoints) == len(imgpoints):
        # Calibrate camera
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            mean_error += error
        
        print( "total error: {}".format(mean_error/len(objpoints)) )
        return mtx, dist,ret
    print("Not enough points for calibration or mismatched number of object and image points.")
    return None, None, None

def undistort_image(img:cv2.typing.MatLike, mtx:cv2.typing.MatLike, dist:cv2.typing.MatLike)->cv2.typing.MatLike:
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    # undistort
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    # crop the image
    return dst

def undistort_and_crop(img:cv2.typing.MatLike, mtx:cv2.typing.MatLike, dist:cv2.typing.MatLike)->tuple[cv2.typing.MatLike, cv2.typing.MatLike]:
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    print("newcameramtx",newcameramtx)
    # undistort
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    return dst,newcameramtx