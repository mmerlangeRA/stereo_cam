
from typing import List
import numpy as np
from scipy.optimize import least_squares
import cv2


def detect_and_compute(image:cv2.typing.MatLike, useAkaze=True):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    descriptor = cv2.AKAZE_create() if useAkaze else cv2.ORB_create()
    keypoints, descriptors = descriptor.detectAndCompute(gray, None)
    return keypoints, descriptors

# Function to detect, compute, and match features between two images
def match_features(images):
    keypoints_list, descriptors_list = zip(*[detect_and_compute(img) for img in images])
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors_list[0], descriptors_list[1])
    matches = sorted(matches, key=lambda x: x.distance)
    return matches, keypoints_list

def reprojection_error(params:List[float],pts1, pts2, K, dist_coeffs:List[float])->float:
    fx = params[0]
    fy = params[1]
    cx= params[2]
    cy= params[3]
    K[0, 0]  = fx
    K[1, 1] = fy
    K[0, 2] = cx
    K[1, 2] = cy
    P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))

    rvec = params[4:7].reshape(3, 1)
    tvec = params[7:10].reshape(3, 1)
    R, _ = cv2.Rodrigues(rvec)
    P2 = K @ np.hstack((R, tvec))

    # Triangulate points
    points_4d = cv2.triangulatePoints(P1, P2, pts1, pts2)
    points_3d = points_4d[:3] / points_4d[3]
    points_3d = points_3d.T
    
    # projected_points= K@points_3d.T 
    # projected_points = projected_points
    # distances = np.abs((pts1 - projected_points).ravel())

    projected_points, _ = cv2.projectPoints(points_3d, rvec, tvec, K, dist_coeffs)
    projected_points = projected_points.reshape(-1, 2)
    distances = np.linalg.norm(pts2.reshape(-1, 2) - projected_points, axis=1)

    distances= distances[distances<30]
    residual = np.average(distances)
    return residual

    # initial_params = np.hstack(([fx_init],[fy_init], [cx_init],[cy_init],rvec_init.ravel(), tvec_init.ravel()))
    # bounds=([100,100,w/2.5, h/2.5,-0.1,-0.1,-0.1,-1.13,-0.03,-0.03],
    #         [2000,2000, w/1.5, h/1.5,0.1,0.1,0.1,-1.11,0.03,0.03])

def compute_auto_calibration_for_images(images:List[cv2.typing.MatLike],verbose=False):
    # Detect and compute features for all images
    keypoints_list, descriptors_list = zip(*[detect_and_compute(img) for img in images])

    # Create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors between the two images
    matches = bf.match(descriptors_list[0], descriptors_list[1])
    matches = sorted(matches, key=lambda x: x.distance)

    if verbose:
    # Draw matches
        img_matches = cv2.drawMatches(images[0], keypoints_list[0], images[1], keypoints_list[1], matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imwrite(f'img_matches.png', img_matches)
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
    focal_length_init = 700.0   #in pixels
    fx_init = fy_init= 700.0   #in pixels
    cx_init = w/2
    cy_init = h/2
    rvec_init = np.array([[-0.03], [0.08], [-0.017]], dtype=np.float32)
    tvec_init = np.array([[-1.12], [0.02], [0.01]], dtype=np.float32)

    K = np.array([[focal_length_init, 0, w / 2],
                [0, focal_length_init, h / 2],
                [0, 0, 1]])


    # Initial parameter guess (focal length)
    initial_params = np.hstack(([fx_init],[fy_init], [cx_init],[cy_init],rvec_init.ravel(), tvec_init.ravel()))
    bounds=([100,100,w/2.5, h/2.5,-0.1,-0.1,-0.1,-1.13,-0.03,-0.03],
            [2000,2000, w/1.5, h/1.5,0.1,0.1,0.1,-1.11,0.03,0.03])

    # Perform bundle adjustment
    result = least_squares(reprojection_error, initial_params, args=(pts1,pts2, K, np.zeros(5)), bounds=bounds)
    refined_params = result.x
    
    refined_params= result.x
    refined_fx = refined_params[0]
    refined_fy = refined_params[1]
    refined_cx = refined_params[2]
    refined_cy = refined_params[3]
    refined_rvec = refined_params[4:7].reshape(3, 1)
    refined_tvec = refined_params[7:10].reshape(3, 1)


    # Update the camera matrix with the refined focal length
    K[0, 0] = refined_fx
    K[1, 1] = refined_fy
    K[0, 2] = refined_cx
    K[1, 2] = refined_cy

    if verbose:
        print("Refined Focal Length: ", refined_fx)
        print("Refined Rotation Vector:\n", refined_rvec)
        print("Refined Translation Vector:\n", refined_tvec)
        print("Updated Camera Matrix:\n", K)

    return K,result.cost,refined_rvec,refined_tvec
