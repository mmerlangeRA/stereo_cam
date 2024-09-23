from random import sample
from typing import List
import numpy as np
from scipy.optimize import minimize
import cv2
from src.features_2d.utils import detectAndComputeKPandDescriptors, getMatches
from src.triangulate.main import get_3d_point_cam1_2_from_coordinates, rotation_matrix_from_params
from src.utils.path_utils import get_static_folder_path

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
        total_residual_in_m = 0.
        for i in range(len(keypoints_cam1)):
            _,_,residual_distance_in_m = get_3d_point_cam1_2_from_coordinates(keypoints_cam1[i], keypoints_cam2[i], image_width, image_height, R, t)
            total_residual_in_m += residual_distance_in_m
        #print("total_residual", total_residual)
        return total_residual_in_m
    result = minimize(optimizeRT, initial_params, bounds=bnds)
    optimized_params = result.x
    total_residual_in_m = result.fun
    return optimized_params,total_residual_in_m

def computeInliersIndices(R, t, keypoints_cam1, keypoints_cam2, reconstruction_threshold, image_width, image_height):
    inliers_indices = []
    mean_distance = 0.
    mean_residual = 0.
    for i in range(len(keypoints_cam1)):
        P1,_,residual_distance_in_m = get_3d_point_cam1_2_from_coordinates(keypoints_cam1[i], keypoints_cam2[i], image_width, image_height, R, t)
        point_distance_in_m =np.linalg.norm(P1)
        if residual_distance_in_m < reconstruction_threshold*point_distance_in_m:
            inliers_indices.append(i)
            mean_residual+=residual_distance_in_m
            mean_distance += point_distance_in_m
    #print(f"mean_distance {mean_distance/len(inliers_indices)}")
    #print(f"mean_residual {mean_residual/len(inliers_indices)}")
    return inliers_indices

def calibrate_left_right(imLeft:cv2.Mat, imRight:cv2.Mat, initial_params,bnds,inlier_threshold,nn_match_ratio=0.5, verbose=False):
    if verbose:
        print("calibrate_left_right")
        print(f"initial_params {initial_params}")
        print(f"bnds {bnds}")
    max_iter = 500
    prob = 0.95
    
    kpts1, desc1 = detectAndComputeKPandDescriptors(imLeft)
    kpts2, desc2 = detectAndComputeKPandDescriptors(imRight)
    
    #kpts to uv
    uv1 = [[k.pt[0], k.pt[1]] for k in kpts1]
    uv2 = [[k.pt[0], k.pt[1]] for k in kpts2]
    good_matches = getMatches(desc1, desc2,nn_match_ratio=nn_match_ratio)
    matched1 = []
    matched2 = []
    for m in good_matches:
        matched1.append(uv1[m.queryIdx])
        matched2.append(uv2[m.trainIdx])


    num_elements = int(len(matched1)*0.5)
    num_elements = 4

    if verbose:
        Matched = cv2.drawMatchesKnn(imLeft, 
                                    kpts1, 
                                    imRight, 
                                    kpts2, 
                                    good_matches, 
                                    outImg=None, 
                                    matchColor=(0, 155, 0), 
                                    singlePointColor=(0, 255, 255), 
                                    matchesMask=good_matches, 
                                    flags=0
                                    ) 
  
        # saving the image  
        cv2.imwrite(get_static_folder_path('Match.jpg'), Matched)
        #cv2.waitKey(0)
        print(f'nb good matches {len(matched1)} ')
    nb_iter = 0
    nb_good_matches = len(matched1)
    best_result = {
        "max_inliers": 0,
        "R":[],
        "t":[],
        "params":[],
    }
    #optimized_params,total_residual_in_m = getCalibrationFrom3Matching(matched1, matched2, initial_params, imLeft.shape[1], imLeft.shape[0],bnds)
    
    while nb_iter<max_iter:
        indices = sample(range(nb_good_matches), num_elements)
        sub_uv1 = [matched1[i] for i in indices]
        sub_uv2 = [matched2[i] for i in indices]
        optimized_params,total_residual_in_m = getCalibrationFrom3Matching(sub_uv1, sub_uv2, initial_params, imLeft.shape[1], imLeft.shape[0],bnds)
        optimized_R = rotation_matrix_from_params(optimized_params[:3])
        optimized_t = optimized_params[3:]

        inliersIndices = computeInliersIndices(optimized_R, optimized_t, matched1, matched2, inlier_threshold, imLeft.shape[1], imLeft.shape[0])
        nb_inliers = len(inliersIndices)
        if nb_inliers ==0:
            continue

        residual_per_num_elements = total_residual_in_m/num_elements
        if nb_inliers > best_result["max_inliers"]:
            best_result["max_inliers"] = nb_inliers
            best_result["R"] = optimized_R
            best_result["params"] = optimized_params
            best_result["t"] = optimized_t
            if nb_inliers == nb_good_matches:
                if verbose:
                    print("all inliers")
                max_iter=-1
            else:
                print(nb_inliers)
                max_iter = np.log(1.-prob)/np.log(1.- pow(nb_inliers/len(matched1),num_elements))
            if verbose:
                print(f'new best result with {nb_inliers} inliers, iteration is {nb_iter}, residual_per_num_elements is {residual_per_num_elements}')
                print("optimized_params", optimized_params)
                print(f"now iter is {nb_iter} and max_iter is {max_iter}")
        nb_iter+=1
    
    refine_results= False

    if refine_results:
        sub_uv1=[matched1[i] for i in inliersIndices]
        sub_uv2=[matched2[i] for i in inliersIndices]
        optimized_params, total_residual_in_m = getCalibrationFrom3Matching(sub_uv1, sub_uv2, initial_params, imLeft.shape[1], imLeft.shape[0],bnds)

        optimized_R = rotation_matrix_from_params(optimized_params[:3])
        optimized_t = optimized_params[3:]
        inliersIndices = computeInliersIndices(optimized_R, optimized_t, matched1, matched2, inlier_threshold, imLeft.shape[1], imLeft.shape[0])
        nb_inliers = len(inliersIndices)
        if verbose:
            print(f'refined best result with {nb_inliers} inliers')
            #best_result["max_inliers"] = nb_inliers
            #best_result["R"] = optimized_R
            #best_result["t"] = optimized_t
            print("refined optimized_params on all inliers", optimized_params)

    return best_result

