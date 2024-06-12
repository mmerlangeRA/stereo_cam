import numpy as np
import cv2 as cv
import yaml
import os

def compute_360_intrinsics(image_width, image_height):
    # Principal point is the center of the image
    cx = image_width / 2
    cy = image_height / 2

    # Approximate focal lengths for equirectangular image
    fx = image_width / (2 * np.pi)
    fy = image_height / np.pi

    # Construct the intrinsic matrix
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]])
    
    return K


def save_calibration(mtx, dist, rmse,fname="calibration_matrix.yaml"):
    data = {'camera_matrix': np.asarray(mtx).tolist(),
        'dist_coeff': np.asarray(dist).tolist(),'rmse':rmse}
    # and save it to a file
    with open(fname, "w") as f:
        yaml.dump(data, f)

def read_calibration(fname="calibration_matrix.yaml"):
    with open(fname, 'r') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return np.asarray(data['camera_matrix']), np.asarray(data['dist_coeff']),data['rmse']


def get_cube_subs(image):
    h,w=image.shape[:2]
    sub_w = int(w/4)
    sub_h = int(h/3)
    #sub_images=[image[:sub_h-1,:sub_w-1],image[sub_h:sub_h*2-1,:sub_w-1],image[sub_h:sub_h*2-1,sub_w:sub_w*2-1],image[sub_h:sub_h*2-1,sub_w*2:sub_w*3-1],image[sub_h:sub_h*2-1,sub_w*3:sub_w*4-1]]
    sub_images=[image[sub_h:sub_h*2-1,:sub_w-1],image[sub_h:sub_h*2-1,sub_w:sub_w*2-1],image[sub_h:sub_h*2-1,sub_w*2:sub_w*3-1],image[sub_h:sub_h*2-1,sub_w*3:sub_w*4-1]]
    #sub_images=[image[sub_h:sub_h*2-1,:sub_w-1]]
    return sub_images

def compute_and_save_calibration(images,chessboard_size,is_cube:bool):
    print(len(images))
    nb_used = 0
    objpoints=[]
    imgpoints=[]
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    for fname in images:
        print(fname)
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        if is_cube:
            sub_images=get_cube_subs(gray)
            for sub in  sub_images:
                th,tw=sub.shape[:2]
                print(f'th={th}, tw={tw}')
                ret, corners = cv.findChessboardCorners(sub, chessboard_size, None)
                # If found, add object points, image points (after refining them)
                if ret == True:
                    #ret, corners = cv.findChessboardCorners(gray, chessboard_size, None)
                    print(f'nb corners={len(corners) if corners is not None else 0}')
                    objpoints.append(objp.copy())
                    corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
                    imgpoints.append(corners2)
                    # Draw and display the corners
                    cv.drawChessboardCorners(gray, chessboard_size, corners2, ret)
                    cv.imshow('img', gray)
                    nb_used+=1
        else:
            print("should not be there if cube")
            # Find the chess board corners
            ret, corners = cv.findChessboardCorners(gray, chessboard_size, None)
            print(f'nb corners={len(corners) if corners is not None else 0}')

            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp.copy())
                corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
                imgpoints.append(corners2)

                # Draw and display the corners
                cv.drawChessboardCorners(img, chessboard_size, corners2, ret)
                cv.imshow('img', img)
                nb_used+=1

    print(f'nb used={nb_used} out of {len(images)}')
    print(f'nb objpoints={len(objpoints)}, nb imgpoints {len(imgpoints)}')
    # Check if we have enough points for calibration
    if len(objpoints) > 0 and len(imgpoints) > 0 and len(objpoints) == len(imgpoints):
        # Calibrate camera
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        save_calibration(mtx,dist,ret, "calibration_matrix.yaml")
        return mtx, dist
    print("Not enough points for calibration or mismatched number of object and image points.")
    return None, None

def undistort(img, mtx, dist):
    h, w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    # undistort
    dst = cv.undistort(img, mtx, dist, None, newcameramtx)
    # crop the image
    return dst

def undistort_and_crop(img, mtx, dist):
    h, w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    # undistort
    dst = cv.undistort(img, mtx, dist, None, newcameramtx)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    return dst