import os
import numpy as np
import cv2 as cv
import yaml
from src.triangulate import get_3d_point_cam1_2_from_coordinates, rotation_matrix_from_params
from src.features import detectAndCompute, getMatches
from scipy.optimize import minimize
from random import randrange,sample

calibration_folder = "calibration"

def save_calibration_params(params, filename):
    np.savetxt(filename+'.csv', params, delimiter=',')

def load_calibration_params(filename):
    loaded =np.loadtxt(filename+'.csv', delimiter=',')
    return loaded

def computeInliers(R, t, keypoints_cam1, keypoints_cam2, threshold, image_width, image_height):
    inliers = []
    for i in range(len(keypoints_cam1)):
        _,_,residual_distance = get_3d_point_cam1_2_from_coordinates(keypoints_cam1[i], keypoints_cam2[i], image_width, image_height, R, t)
        if residual_distance < threshold:
            inliers.append(i)
    return inliers


def getCalibrationFrom3Matching(keypoints_cam1,keypoints_cam2, initial_params, image_width, image_height,bnds,verbose=False):
    if len(keypoints_cam1) ==0:
        raise Exception("No matches found")
    
    if len(keypoints_cam1) != len(keypoints_cam2):
        raise Exception("Inconsistent nb of matches")
    
    def optimizeRT(params):
        R = rotation_matrix_from_params(params[:3])
        t = params[3:]
        if verbose:
            print("getCalibrationFrom3Matching")
            print("R",R)
            print("t", t)
            print("keypoints_cam1",keypoints_cam1)
        total_residual = 0.
        for i in range(len(keypoints_cam1)):
            _,_,residual_distance_normalized = get_3d_point_cam1_2_from_coordinates(keypoints_cam1[i], keypoints_cam2[i], image_width, image_height, R, t)
            total_residual += residual_distance_normalized
        #print("total_residual",total_residual)
        return total_residual
    result = minimize(optimizeRT, initial_params, bounds=bnds)
    optimized_params = result.x
    residual_distance_normalized = result.fun / len(keypoints_cam1)
    return optimized_params,residual_distance_normalized


def calibrate_left_right(imLeft:cv.Mat, imRight:cv.Mat, initial_params,bnds,inlier_threshold,nn_match_ratio=0.5):
    print("calibrate_left_right")
    print(f"initial_params {initial_params}")
    print(f"bnds {bnds}")
    max_iter = 500
    prob = 0.95
    num_elements = 4
    kpts1, desc1 = detectAndCompute(imLeft)
    kpts2, desc2 = detectAndCompute(imRight)
    #kpts to uv
    print(kpts1[0].pt)
    uv1 = [[k.pt[0], k.pt[1]] for k in kpts1]
    uv2 = [[k.pt[0], k.pt[1]] for k in kpts2]
    nn_matches = getMatches(desc1, desc2)
    matched1 = []
    matched2 = []
    good_matches = [[0, 0] for i in range(len(nn_matches))] 
    for i, (m, n) in enumerate(nn_matches):
        if m.distance < nn_match_ratio * n.distance:
            matched1.append(uv1[m.queryIdx])
            matched2.append(uv2[m.trainIdx])
            good_matches[i] = [1, 0] 
    Matched = cv.drawMatchesKnn(imLeft, 
                             kpts1, 
                             imRight, 
                             kpts2, 
                             nn_matches, 
                             outImg=None, 
                             matchColor=(0, 155, 0), 
                             singlePointColor=(0, 255, 255), 
                             matchesMask=good_matches, 
                             flags=0
                             ) 
  
# Displaying the image  
    cv.imwrite('Match.jpg', Matched)
    #cv.waitKey(0)
    print(f'nb good matches {len(matched1)} out of {len(nn_matches)}')
    nb_iter = 0
    nb_good_matches = len(matched1)
    best_result = {
        "max_inliers": 0,
        "R":[],
        "t":[],
        "params":[],
    }
    while nb_iter<max_iter:
        indices = sample(range(nb_good_matches), num_elements)
        sub_uv1 = [matched1[i] for i in indices]
        sub_uv2 = [matched2[i] for i in indices]
        optimized_params,residual = getCalibrationFrom3Matching(sub_uv1, sub_uv2, initial_params, imLeft.shape[1], imLeft.shape[0],bnds)
        optimized_R = rotation_matrix_from_params(optimized_params[:3])
        optimized_t = optimized_params[3:]

        inliers = computeInliers(optimized_R, optimized_t, matched1, matched2, inlier_threshold, imLeft.shape[1], imLeft.shape[0])
        nb_inliers = len(inliers)
        if nb_inliers ==0:
            continue

        residual_per_inlier = residual/nb_inliers
        if nb_inliers > best_result["max_inliers"]:
            print(f'new best result with {nb_inliers} inliers, iteration is {nb_iter}, residual_per_inlier is {residual_per_inlier}')
            print("optimized_params", optimized_params)
            best_result["max_inliers"] = nb_inliers
            best_result["R"] = optimized_R
            best_result["params"] = optimized_params
            best_result["t"] = optimized_t
            max_iter = np.log(1.-prob)/np.log(1.- pow(nb_inliers/len(matched1),num_elements))
            print(f"now iter is {nb_iter} and max_iter is {max_iter}")
        nb_iter+=1
    
    sub_uv1=[matched1[i] for i in inliers]
    sub_uv2=[matched2[i] for i in inliers]
    optimized_params, residual = getCalibrationFrom3Matching(sub_uv1, sub_uv2, initial_params, imLeft.shape[1], imLeft.shape[0],bnds)

    optimized_R = rotation_matrix_from_params(optimized_params[:3])
    optimized_t = optimized_params[3:]
    inliers = computeInliers(optimized_R, optimized_t, matched1, matched2, inlier_threshold, imLeft.shape[1], imLeft.shape[0])
    nb_inliers = len(inliers)
    print(f'refined best result with {nb_inliers} inliers')
    #best_result["max_inliers"] = nb_inliers
    #best_result["R"] = optimized_R
    #best_result["t"] = optimized_t
    print("refined optimized_params on all inliers", optimized_params)

    return best_result


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
        save_calibration_custom(mtx,dist,ret, "calibration_matrix.yaml")
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