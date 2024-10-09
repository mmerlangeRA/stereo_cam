from typing import List, Tuple
import cv2
import numpy as np
from src.features_2d.utils import detectAndComputeKPandDescriptors
from scipy.optimize import least_squares

from src.utils.path_utils import get_output_path

def compute_reprojection_residual(params:List[float],pts1, pts2, dist_coeffs:List[float])->float:
    """
    Computes the reprojection error for optimization.

    Args:
        params (np.ndarray): Array containing camera parameters (fx, fy, cx, cy, rvec, tvec).
        pts1 (np.ndarray): Array of points from the first image.
        pts2 (np.ndarray): Array of points from the second image.
        K (np.ndarray): Camera matrix.
        dist_coeffs (np.ndarray): Distortion coefficients.

    Returns:
        np.ndarray: Reprojection error.
    """

    # Ensure pts1 and pts2 are in 2xN format
    # pts1 = pts1.T if pts1.shape[0] != 2 else pts1
    # pts2 = pts2.T if pts2.shape[0] != 2 else pts2

    fx = params[0]
    fy = params[1]
    cx= params[2]
    cy= params[3]
    K = np.array([[fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]])
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))

    rvec = params[4:7].reshape(3, 1)
    tvec = params[7:10].reshape(3, 1)
    R, _ = cv2.Rodrigues(rvec)
    P2 = K @ np.hstack((R, tvec))

    # Triangulate points
    points_4d = cv2.triangulatePoints(P1, P2, pts1, pts2)

    points_3d = points_4d[:3] / points_4d[3]
    points_3d = points_3d.T

    projected_points, _ = cv2.projectPoints(points_3d, rvec, tvec, K, dist_coeffs)
    projected_points = projected_points.reshape(-1, 2)
    distances = np.linalg.norm(pts2.reshape(-1, 2) - projected_points, axis=1)

    distances= distances[distances<30]
    residual = np.average(distances)
    return residual

def compute_auto_calibration_for_2_stereo_standard_images(imgLeft:cv2.typing.MatLike, imgRight:cv2.typing.MatLike,verbose=True)-> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Automatically calibrates a stereo camera setup using two standard images.

    Args:
        imgLeft (cv_typing.MatLike): Left stereo image.
        imgRight (cv_typing.MatLike): Right stereo image.
        verbose (bool): If True, prints and saves intermediate results. Default is False.

    Returns:
        Tuple[np.ndarray, float, np.ndarray, np.ndarray]: 
            - K: Refined camera matrix.
            - cost: Final reprojection error cost.
            - refined_rvec: Refined rotation vector.
            - refined_tvec: Refined translation vector.
    """
    images=[imgLeft,imgRight]
    # Detect and compute features for all images
    keypoints_list, descriptors_list = zip(*[detectAndComputeKPandDescriptors(img) for img in images])

    # Create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors between the two images
    matches = bf.match(descriptors_list[0], descriptors_list[1])
    matches = sorted(matches, key=lambda x: x.distance)

    if verbose:
    # Draw matches
        img_matches = cv2.drawMatches(images[0], keypoints_list[0], images[1], keypoints_list[1], matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imwrite(get_output_path('img_matches.png'), img_matches)
    # plt.imshow(img_matches)
    # plt.show()

    # Extract matched keypoints
    pts1 = np.float32([keypoints_list[0][m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([keypoints_list[1][m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    y_threshold = 30.0  # Adjust this threshold as needed

    # Filter out points with y-position differences greater than the threshold
    filtered_pts1 = []
    filtered_pts2 = []

    for p1, p2 in zip(pts1, pts2):
        if abs(p1[0][1] - p2[0][1]) <= y_threshold:
            filtered_pts1.append(p1)
            filtered_pts2.append(p2)

    # Convert lists back to numpy arrays
    pts1 = np.array(filtered_pts1).reshape(-1, 1, 2)
    pts2 = np.array(filtered_pts2).reshape(-1, 1, 2)

    h, w = images[0].shape[:2]

    # Initial guess 
    fx_init = fy_init= 700.0   #in pixels
    cx_init = w/2
    cy_init = h/2
    rvec_init = np.array([[-0.03], [0.08], [-0.017]], dtype=np.float32)
    tvec_init = np.array([[-1.12], [0.02], [0.01]], dtype=np.float32)


    # Initial parameter guess (focal length)
    initial_params = np.hstack(([fx_init],[fy_init], [cx_init],[cy_init],rvec_init.ravel(), tvec_init.ravel()))
    bounds=([100,100,w/2.5, h/2.5,-0.1,-0.1,-0.1,-1.13,-0.03,-0.03],
            [2000,2000, w/1.5, h/1.5,0.1,0.1,0.1,-1.11,0.03,0.03])

    # Perform bundle adjustment
    result = least_squares(compute_reprojection_residual, initial_params, args=(pts1,pts2, np.zeros(5)), bounds=bounds, loss='huber')
    refined_params = result.x
    
    refined_params= result.x
    refined_fx = refined_params[0]
    refined_fy = refined_params[1]
    refined_cx = refined_params[2]
    refined_cy = refined_params[3]
    refined_rvec = refined_params[4:7].reshape(3, 1)
    refined_tvec = refined_params[7:10].reshape(3, 1)


    # Update the camera matrix with the refined focal length
    K = np.array([[refined_fx, 0, refined_cx],
            [0, refined_fx, refined_cy],
            [0, 0, 1]])

    if verbose:
        print("nb kp",len(filtered_pts1))
        print("Refined Focal Length: ", refined_fx)
        print("Refined Rotation Vector:\n", refined_rvec)
        print("Refined Translation Vector:\n", refined_tvec)
        print("Updated Camera Matrix:\n", K)
        print("Final Reprojection Error Cost:", result.cost)

    return K,result.cost,refined_rvec,refined_tvec
