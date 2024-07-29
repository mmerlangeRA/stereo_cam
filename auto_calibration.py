import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt

from src.calibrate_Cube import get_cube_front

# Load images
cube_images_paths = glob.glob('auto_cube/*.png')

cube_images_paths=[]
for index in range(2):
    for sub in range(3):
        cube_images_paths.append(f'auto_cube/D_P{index+1}_CAM_G_{sub}_CUBE.png')
        cube_images_paths.append(f'auto_cube/D_P{index+1}_CAM_D_{sub}_CUBE.png')
print(cube_images_paths)
#cube_images_paths = ['auto_cube/D_P1_CAM_G_1_CUBE.png','auto_cube/D_P1_CAM_D_1_CUBE.png']
cube_images = [cv2.imread(img_path) for img_path in cube_images_paths]
images=[get_cube_front(img) for img in cube_images] 

# Create ORB detector
orb = cv2.ORB_create()

# Function to detect and compute features
def detect_and_compute(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    return keypoints, descriptors

def bundle_adjustment(points_3d, points_2d, camera_matrix, dist_coeffs, rvec, tvec):
    for i in range(10):  # Iterate 10 times for refinement
        # Project 3D points to 2D
        projected_points, _ = cv2.projectPoints(points_3d, rvec, tvec, camera_matrix, dist_coeffs)
        projected_points = projected_points.reshape(-1, 2)
        
        # Compute reprojection error
        error = np.linalg.norm(points_2d - projected_points, axis=1).mean()
        print(f"Iteration {i+1}, Reprojection Error: {error}")
        
        # Refine pose using solvePnP
        _, rvec, tvec = cv2.solvePnP(points_3d, points_2d, camera_matrix, dist_coeffs, rvec, tvec, useExtrinsicGuess=True)

    return rvec, tvec

def compute_for_images(images):
    # Detect and compute features for all images
    keypoints_list, descriptors_list = zip(*[detect_and_compute(img) for img in images])

    # Create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors between the two images
    matches = bf.match(descriptors_list[0], descriptors_list[1])
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw matches
    img_matches = cv2.drawMatches(images[0], keypoints_list[0], images[1], keypoints_list[1], matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img_matches)
    plt.show()

    # Extract matched keypoints
    pts1 = np.float32([keypoints_list[0][m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([keypoints_list[1][m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Find the essential matrix
    E, mask = cv2.findEssentialMat(pts1, pts2, method=cv2.RANSAC, prob=0.999, threshold=1.0)

    # Select inlier points
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]

    # Recover pose from the essential matrix
    _, R, t, mask = cv2.recoverPose(E, pts1, pts2)

    print("Rotation Matrix:\n", R)
    print("Translation Vector:\n", t)

    # Get the camera matrix (assuming fx = fy and principal point at the center)
    h, w = images[0].shape[:2]
    focal_length = 1.0  # Initial guess
    K = np.array([[focal_length, 0, w / 2],
                [0, focal_length, h / 2],
                [0, 0, 1]])

    # Projection matrices for the two views
    P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = np.hstack((R, t))

    # Triangulate points
    points_4d = cv2.triangulatePoints(P1, P2, pts1, pts2)
    points_3d = points_4d[:3] / points_4d[3]
    points_3d = points_3d.T

    # Initial pose estimation using solvePnPRansac
    _, rvec, tvec, inliers = cv2.solvePnPRansac(points_3d, pts1, K, None)

    # Refine pose
    rvec, tvec = bundle_adjustment(points_3d, pts1, K, None, rvec, tvec)

    print("Refined Rotation Vector:\n", rvec)
    print("Refined Translation Vector:\n", tvec)

    # Recalculate the focal length using the refined pose
    projected_points, _ = cv2.projectPoints(points_3d, rvec, tvec, K, None)
    projected_points = projected_points.reshape(-1, 2)
    focal_lengths = []

    for pt2d, pt3d in zip(projected_points, points_3d):
        z = pt3d[2]
        if z != 0:
            fx = pt2d[0] / z
            fy = pt2d[1] / z
            focal_lengths.append((fx + fy) / 2)

    focal_length = np.mean(focal_lengths)
    print("Estimated Focal Length: ", focal_length)
    return focal_length

focal_lengths = []
for i in range(0,len(images),2):
    focal_length =compute_for_images(images[i:i+2])
    focal_lengths.append(focal_length)

print(focal_lengths)