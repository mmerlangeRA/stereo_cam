from random import sample
import time
from typing import List, Tuple
import numpy as np
from scipy.optimize import minimize
import cv2

from src.features_2d.utils import AkazeDescriptorManager, OrbDescriptorManager, detectAndComputeKPandDescriptors, detectAndComputeKPandDescriptors_new, getMatches
from src.triangulate.main import get_3d_point_cam1_2_from_coordinates, rotation_matrix_from_params
from src.utils.coordinate_transforms import cartesian_to_equirectangular, pixel_to_spherical, spherical_to_cartesian
from src.utils.path_utils import get_ouput_path
from src.utils.TransformClass import TransformBounds, Transform
from src.road_detection.common import AttentionWindow

def getRefinedTransformFromKPMatching(
        keypoints_cam1:np.ndarray,keypoints_cam2:np.ndarray,
        estimatedTransform:Transform,
        image_width:int, image_height:int,
        transfromBounds:TransformBounds,verbose=False)->Tuple[Transform,float]:
    
    initial_params = estimatedTransform.as_params()
    baseline = estimatedTransform.xc
    bnds= transfromBounds.get_bounds()

    if len(keypoints_cam1) ==0:
        raise Exception("No matches found")
    
    if len(keypoints_cam1) != len(keypoints_cam2):
        raise Exception("Inconsistent nb of matches")
    
    def optimizeRT(params):
        R = rotation_matrix_from_params(params[2:])
        t = np.array([baseline] + list(params[:2]))
        if verbose:
            print("getRefinedTransformFrom3Matching")
            print("R",R)
            print("t", t)
            print("keypoints_cam1",keypoints_cam1)
        total_residual_in_m = 0.
        _,_,residual_distance_in_m = get_3d_point_cam1_2_from_coordinates(keypoints_cam1, keypoints_cam2, image_width, image_height, R, t)
        total_residual_in_m = np.sum(residual_distance_in_m)
        #print("total_residual", total_residual)
        return total_residual_in_m
    
    result = minimize(optimizeRT, initial_params, bounds=bnds)
    optimized_params = result.x
    total_residual_in_m = result.fun
    optimized_params=np.array([baseline] + list(optimized_params))
    refined_transform = Transform(*optimized_params)
    #refined_transform = Transform(*optimized_params)
    return refined_transform,total_residual_in_m


def computeInliersIndices(
    R: np.ndarray, 
    t: np.ndarray, 
    keypoints_cam1: np.ndarray, 
    keypoints_cam2: np.ndarray, 
    reconstruction_threshold: float, 
    image_width: int, 
    image_height: int
) -> List[int]:
    """
    Compute inlier indices based on 3D point reconstruction residuals.

    Parameters:
    - R: Rotation matrix between cameras.
    - t: Translation vector between cameras.
    - keypoints_cam1: Array of keypoints from camera 1.
    - keypoints_cam2: Array of keypoints from camera 2.
    - reconstruction_threshold: Threshold for considering points as inliers.
    - image_width: Width of the equirectangular image.
    - image_height: Height of the equirectangular image.

    Returns:
    - inliers_indices: List of indices of keypoints that are considered inliers.
    """
    
    # Compute 3D points and residuals for all keypoints in one go
    P1, P2, residual_distance_in_m = get_3d_point_cam1_2_from_coordinates(
        keypoints_cam1, keypoints_cam2, image_width, image_height, R, t, verbose=False)
    
    #get rays
    method_screen = False
    if method_screen:
    
        u,v=cartesian_to_equirectangular(P2[:,0],P2[:,1],P2[:,2],image_width=image_width,image_height=image_height,to_int=False)
        # Extract keypoints from cam1
        keypoints_u = keypoints_cam1[:, 0]  # Extract x-coordinates of keypoints
        keypoints_v = keypoints_cam1[:, 1]  # Extract y-coordinates of keypoints

        # Calculate the maximum distance for each point in a vectorized way
        distances = np.maximum(np.abs(u - keypoints_u), np.abs(v - keypoints_v))

        # Check where distances are less than the threshold (3)

        inliers_mask = distances < 1.2
    else:
        # Compute Euclidean distance for all 3D points (P1) in one go
        point_distances_in_m = np.linalg.norm(P1, axis=1)
        # Use NumPy to create a boolean mask for inliers
        inliers_mask = residual_distance_in_m < reconstruction_threshold * point_distances_in_m
        #debugging purposes. Delete if needed
        distances = residual_distance_in_m/(reconstruction_threshold * point_distances_in_m)
    
    # Extract inlier indices efficiently
    inliers_indices = np.where(inliers_mask)[0].tolist()
    
    # Calculate the mean distance and mean residual for inliers
    #mean_distance = np.mean(point_distances_in_m[inliers_mask]) if inliers_indices else 0.
    #mean_residual = np.mean(residual_distance_in_m[inliers_mask]) if inliers_indices else 0.
    
    # Print mean values if needed (commented out for now)
    # print(f"mean_distance: {mean_distance}")
    # print(f"mean_residual: {mean_residual}")

    return inliers_indices, distances


def auto_compute_cam2_transform(imLeft:cv2.Mat, imRight:cv2.Mat, estimatedTransform: Transform,transformBounds:TransformBounds,inlier_threshold:float,attention_window:AttentionWindow=None,  verbose=False)->Tuple[Transform, float]:
    if verbose:
        print("auto_compute_cam2_transform")
        print(f"estimatedTransform {estimatedTransform}")
        print(f"bnds {transformBounds}")

    times={
        "pre":0,
        "getRefinedTransformFromKPMatching":0,
        "inliers":0,
        "post":0
    }

    detectorManager = OrbDescriptorManager()
    detectorManager = AkazeDescriptorManager()
    #num_elements = int(len(matched1)*0.5)
    num_elements = 3
    max_iter = 500
    prob = 0.95
    start = time.time()

    if False:
        kpts1, desc1 = detectorManager.detectAndComputeKPandDescriptors_zone(imLeft,top_limit=topLimit,bottom_limit=bottomLimit,wished_nb_kpts=4000)
        kpts2, desc2 = detectorManager.detectAndComputeKPandDescriptors_zone(imRight,top_limit=topLimit,bottom_limit=bottomLimit,wished_nb_kpts=4000)
    else:
        kpts1, desc1 = detectorManager.detectAndComputeKPandDescriptors(imLeft,attentionWindow=attention_window)
        kpts2, desc2 = detectorManager.detectAndComputeKPandDescriptors(imRight,attentionWindow=attention_window)
    
    #kpts to uv
    uv1 = np.array([[k.pt[0], k.pt[1]] for k in kpts1])
    uv2 = np.array([[k.pt[0], k.pt[1]] for k in kpts2])

    good_matches = detectorManager.getMatches(desc1, desc2)
    print("nb keypoints",len(kpts1))
    print(f'nb good matches {len(good_matches)} ')
    #good_matches = good_matches[:100]
    
    query_indices = [m.queryIdx for m in good_matches]
    train_indices = [m.trainIdx for m in good_matches]

    # Use NumPy indexing to extract the corresponding matched points
    matched1 = np.array(uv1[query_indices])  # Points from camera 1
    matched2 = np.array(uv2[train_indices])  # Points from camera 2

    if verbose:
        good_matches_knn = [[m] for m in good_matches]
        Matched = cv2.drawMatchesKnn(imLeft, 
                                    kpts1, 
                                    imRight, 
                                    kpts2, 
                                    good_matches_knn, 
                                    outImg=None, 
                                    matchColor=(0, 155, 0), 
                                    singlePointColor=(0, 255, 255), 
                                    matchesMask=None, 
                                    flags=0
                                    ) 
  
        # saving the image  
        cv2.imwrite(get_ouput_path('Match.jpg'), Matched)
        #cv2.waitKey(0)
        print(f'nb good matches {len(matched1)} ')
    nb_iter = 0
    nb_good_matches = len(matched1)
    best_nb_inliers = 0
    best_transform:Transform = None
    endProcessing = time.time()
    times["pre"] = endProcessing-start
    time_in_getRefined=0
    time_in_inliers=0
    time_in_post=0
    while nb_iter<max_iter:
        indices = sample(range(nb_good_matches), num_elements)
        sub_uv1 = np.array([matched1[i] for i in indices])
        sub_uv2 = np.array([matched2[i] for i in indices])
        startRefine  = time.time()
        refine_transform,total_residual_in_m = getRefinedTransformFromKPMatching(
            sub_uv1, sub_uv2, 
            estimatedTransform=estimatedTransform, 
            image_width=imLeft.shape[1], image_height=imLeft.shape[0], transfromBounds=transformBounds)

        optimized_R = refine_transform.rotationMatrix
        optimized_t = refine_transform.translationVector
        startInliers = time.time()
        time_in_getRefined += startInliers-startRefine
        inliersIndices,distances = computeInliersIndices(optimized_R, optimized_t, matched1, matched2, inlier_threshold, imLeft.shape[1], imLeft.shape[0])
        nb_inliers = len(inliersIndices)
        endInliers = time.time()
        time_in_inliers += endInliers-startInliers
        if nb_inliers ==0:
            continue

        residual_per_num_elements = total_residual_in_m/num_elements
        if nb_inliers > best_nb_inliers:
            best_transform = refine_transform
            best_nb_inliers = nb_inliers
            if nb_inliers == nb_good_matches:
                if verbose:
                    print("all inliers")
                max_iter=-1
            else:
                #update the number of iterations
                max_iter= min(max_iter,np.log(1.-prob)/np.log(1.- pow(nb_inliers/nb_good_matches,num_elements)))
            if verbose:
                debug_image = imLeft.copy()
                for i in range(len(matched1)):
                    clipValue = 3.
                    distances=np.clip(np.array(distances), None, clipValue)
                    d= distances[i]/clipValue
                    color = (0., int(255*(1.-d)),int(255*d)) 
                    cv2.circle(debug_image, (int(matched1[i][0]), int(matched1[i][1])), 10, color, -1)
                d_path=get_ouput_path(f"debug{nb_iter}.png")
                cv2.imwrite(d_path, debug_image)

                print(f'new best result with {nb_inliers} inliers, iteration is {nb_iter}, residual_per_num_elements is {residual_per_num_elements}')
                print("optimized_params", refine_transform)
                print(f"now iter is {nb_iter} and max_iter is {max_iter}")
        nb_iter+=1
    
    refine_results= True

    if refine_results:
        sub_uv1=np.array([matched1[i] for i in inliersIndices])
        sub_uv2=np.array([matched2[i] for i in inliersIndices])
        refine_transform, total_residual_in_m = getRefinedTransformFromKPMatching(sub_uv1, sub_uv2, 
                                                                                  estimatedTransform=estimatedTransform, 
                                                                                  image_width= imLeft.shape[1], image_height= imLeft.shape[0],
                                                                                  transfromBounds=transformBounds)
        optimized_R = refine_transform.rotationMatrix
        optimized_t = refine_transform.translationVector
        inliersIndices,_ = computeInliersIndices(optimized_R, optimized_t, matched1, matched2, inlier_threshold, imLeft.shape[1], imLeft.shape[0])
        nb_inliers = len(inliersIndices)
        if verbose:
            print(f'refined best result with {nb_inliers} inliers vs {best_nb_inliers}')
        if nb_inliers > best_nb_inliers:
            best_transform = refine_transform
            best_nb_inliers=nb_inliers
            if verbose:
                
                #best_result["max_inliers"] = nb_inliers
                #best_result["R"] = optimized_R
                #best_result["t"] = optimized_t
                print("refined optimized_params on all inliers", refine_transform)

    times["getRefinedTransformFromKPMatching"]=time_in_getRefined
    times["inliers"]=time_in_inliers
    times["post"]=time_in_post
    times["total"]=time.time()-start

    if verbose:
        print(times)

    return best_transform,best_nb_inliers/nb_good_matches

def compute_stereo_matched_KP(imLeft:cv2.Mat, imRight:cv2.Mat, camRightTransform: Transform,inlier_threshold:float,attention_window:AttentionWindow,  mask_left, mask_right, verbose=False)->Tuple[Transform, float]:
    if verbose:
        print("auto_compute_cam2_transform")
        print(f"camRightTransform {camRightTransform}")

    times={
        "pre":0,
        "getRefinedTransformFromKPMatching":0,
        "inliers":0,
        "post":0
    }

    detectorManager = OrbDescriptorManager(nfeatures=5000)
    #detectorManager = AkazeDescriptorManager()

    start = time.time()

    kpts1, desc1 = detectorManager.detectAndComputeKPandDescriptors(imLeft,mask=mask_left)
    kpts2, desc2 = detectorManager.detectAndComputeKPandDescriptors(imRight,mask=mask_right)
    
    #kpts to uv
    uv1 = np.array([[k.pt[0], k.pt[1]] for k in kpts1])
    uv2 = np.array([[k.pt[0], k.pt[1]] for k in kpts2])

    good_matches = detectorManager.getMatches(desc1, desc2)
    print("nb keypoints",len(kpts1))
    print(f'nb good matches {len(good_matches)} ')
    #good_matches = good_matches[:100]
    
    query_indices = [m.queryIdx for m in good_matches]
    train_indices = [m.trainIdx for m in good_matches]

    # Use NumPy indexing to extract the corresponding matched points
    matched1 = np.array(uv1[query_indices])  # Points from camera 1
    matched2 = np.array(uv2[train_indices])  # Points from camera 2

    if verbose:
        good_matches_knn = [[m] for m in good_matches]
        Matched = cv2.drawMatchesKnn(imLeft, 
                                    kpts1, 
                                    imRight, 
                                    kpts2, 
                                    good_matches_knn, 
                                    outImg=None, 
                                    matchColor=(0, 155, 0), 
                                    singlePointColor=(0, 255, 255), 
                                    matchesMask=None, 
                                    flags=0
                                    ) 
  
        # saving the image  
        cv2.imwrite(get_ouput_path('Match.jpg'), Matched)
        #cv2.waitKey(0)
        print(f'nb good matches {len(matched1)} ')

    endProcessing = time.time()
    times["pre"] = endProcessing-start
    time_in_getRefined=0
    time_in_inliers=0
    time_in_post=0
    optimized_R = camRightTransform.rotationMatrix
    optimized_t = camRightTransform.translationVector
    image_height,image_width=imLeft.shape[:2]
    P1, P2, residual_distance_in_m = get_3d_point_cam1_2_from_coordinates(
        matched1, matched2, image_width, image_height, optimized_R, optimized_t, verbose=False)
    inliersIndices,_ = computeInliersIndices(optimized_R, optimized_t, matched1, matched2, inlier_threshold, imLeft.shape[1], imLeft.shape[0])
    point_distances_in_m = np.linalg.norm(P1, axis=1)
    # Use NumPy to create a boolean mask for inliers
    inliers_mask = residual_distance_in_m < inlier_threshold * point_distances_in_m
    #debugging purposes. Delete if needed
    distances = residual_distance_in_m/(inlier_threshold * point_distances_in_m)
    
    # Extract inlier indices efficiently
    inliersIndices = np.where(inliers_mask)[0].tolist()
    sub_uv1=np.array([matched1[i] for i in inliersIndices])
    sub_uv2=np.array([matched2[i] for i in inliersIndices])
    P1=np.array([P1[i] for i in inliersIndices])
    P2=np.array([P2[i] for i in inliersIndices])

    debug_image = imLeft.copy()
    for p in sub_uv1:
        cv2.circle(debug_image, (int(p[0]), int(p[1])), 2, (0,255,0), -1)
    d_path=get_ouput_path(f"debug_road.png")
    cv2.imwrite(d_path, debug_image)
        

    times["getRefinedTransformFromKPMatching"]=time_in_getRefined
    times["inliers"]=time_in_inliers
    times["post"]=time_in_post
    times["total"]=time.time()-start

    if verbose:
        print(times)

    return sub_uv1,sub_uv2,P1,P2

