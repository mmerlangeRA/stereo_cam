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


def compute_cube_calibration(image_paths:List[str],chessboard_size:cv2.typing.Size,square_size:float, use_only_front=True,verbose=False)->tuple[cv2.typing.MatLike, cv2.typing.MatLike,float]:
    """
    Calibrates a camera using images of a chessboard pattern extracted from cube images.

    Parameters:
        image_paths (List[str]): List of file paths to the cube images.
        chessboard_size (Tuple[int, int]): Number of inner corners per a chessboard row and column (rows, columns).
        square_size (float): Size of a square in your defined unit (e.g., meters).
        verbose (bool, optional): If True, prints detailed information during processing. Defaults to False.

    Returns:
        Tuple[np.ndarray, np.ndarray, float]: Returns the camera matrix (mtx), distortion coefficients (dist), and the RMS re-projection error (ret).

    Raises:
        Exception: If not enough points are found for calibration or if there is a mismatch in the number of object and image points.
    """
    nb_used = 0
    objpoints=[]
    imgpoints=[]
    rows, cols = chessboard_size
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)
    objp *= square_size

    # objp2 = np.zeros((6*9,3), np.float32)
    # objp2[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
    # objp2 *= square_size 
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    window_size = (11, 11)
    zero_zone = (-1, -1)

    for fname in image_paths:
        img = cv2.imread(fname)
        if use_only_front:  
            cube_faces=[get_cube_front_image(img)]
        else:
            cube_faces = get_cube_sub_images(img)
        for face_img in cube_faces:
        #front_image=get_cube_front_image(img)
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            th,tw=gray.shape[:2]
            if verbose:
                print(fname)
                print(f'th={th}, tw={tw}')
            ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp.copy())

                corners2 = cv2.cornerSubPix(gray, corners, window_size, zero_zone, criteria)
                imgpoints.append(corners2)
                if verbose:
                    print(f'nb corners={len(corners) if corners is not None else 0}')
                    # Draw and display the corners
                    cv2.drawChessboardCorners(gray, chessboard_size, corners2, ret)
                    cv2.imshow('img', gray)
                    cv2.waitKey(500)
                nb_used+=1

    if verbose:
        print(f'nb used={nb_used} out of {len(image_paths)}')
        print(f'nb objpoints={len(objpoints)}, nb imgpoints {len(imgpoints)}')
    # Check if we have enough points for calibration
    if len(objpoints) > 0 and len(imgpoints) > 0 and len(objpoints) == len(imgpoints):
        # Calibrate camera
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        total_error = 0
        total_points = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)
            total_error += error**2
            total_points += len(imgpoints[i])
        mean_square_error = np.sqrt(total_error / total_points)
        
        if verbose:
            print( "mean_square_error error: {}".format(mean_square_error) )
        return mtx, dist,ret
    raise ValueError(f"Calibration failed: Not enough points. Object points: {len(objpoints)}, Image points: {len(imgpoints)}")

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