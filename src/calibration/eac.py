from random import sample
from typing import List
import numpy as np
from scipy.optimize import minimize
import cv2
from src.features_2d.utils import detectAndCompute, getMatches
from src.triangulate.main import get_3d_point_cam1_2_from_coordinates, rotation_matrix_from_params




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

def computeInliers(R, t, keypoints_cam1, keypoints_cam2, threshold, image_width, image_height):
    inliers = []
    for i in range(len(keypoints_cam1)):
        _,_,residual_distance = get_3d_point_cam1_2_from_coordinates(keypoints_cam1[i], keypoints_cam2[i], image_width, image_height, R, t)
        if residual_distance < threshold:
            inliers.append(i)
    return inliers

def calibrate_left_right(imLeft:cv2.Mat, imRight:cv2.Mat, initial_params,bnds,inlier_threshold,nn_match_ratio=0.5):
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
    Matched = cv2.drawMatchesKnn(imLeft, 
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
    cv2.imwrite('Match.jpg', Matched)
    #cv2.waitKey(0)
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

