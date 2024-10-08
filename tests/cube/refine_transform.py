from typing import Tuple
import cv2
import numpy as np

from scipy.optimize import least_squares

from src.features_2d.utils import detectAndComputeKPandDescriptors, getMatches

def equirectangular_to_cubemap(img:cv2.typing.MatLike, face_size=1024):
    """
    Convert an equirectangular image to a cubemap.
    
    Parameters:
        img (np.ndarray): Input equirectangular image.
        face_size (int): Size of each cube face.
    
    Returns:
        dict: A dictionary of cube faces with keys ['front', 'back', 'left', 'right', 'top', 'bottom'].
    """
    # Define cube faces and their corresponding rotation matrices
    faces = {
        'front': {'center': [0, 0, -1], 'up': [0, -1, 0]},
        'back': {'center': [0, 0, 1], 'up': [0, -1, 0]},
        'left': {'center': [1, 0, 0], 'up': [0, -1, 0]},
        'right': {'center': [-1, 0, 0], 'up': [0, -1, 0]},
        'top': {'center': [0, 1, 0], 'up': [0, 0, 1]},
        'bottom': {'center': [0, -1, 0], 'up': [0, 0, -1]},
    }

    h, w = img.shape[:2]
    faces_imgs = {}

    for face in faces:
        center = np.array(faces[face]['center'], dtype=np.float32)
        up = np.array(faces[face]['up'], dtype=np.float32)
        # Compute rotation matrix
        rot_matrix = compute_rotation_matrix(center, up)
        # Generate the mapping
        map_x, map_y = generate_cubemap_face_mapping(face_size, rot_matrix, h, w)
        # Remap the image
        face_img = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        faces_imgs[face] = face_img

    return faces_imgs

def compute_rotation_matrix(forward, up):
    """
    Compute a rotation matrix given forward and up vectors.
    """
    forward = forward / np.linalg.norm(forward)
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    rot_matrix = np.array([right, up, forward], dtype=np.float32)
    return rot_matrix

def generate_cubemap_face_mapping(face_size, rot_matrix, h, w):
    """
    Generate mapping coordinates for a cubemap face.
    """
    # Reverse the u axis to fix horizontal flipping
    u = np.linspace(-1, 1, face_size)
    v = np.linspace(1, -1, face_size)
    uu, vv = np.meshgrid(u, v)
    direction = np.stack([uu, vv, -np.ones_like(uu)], axis=-1)
    direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
    direction = direction.reshape(-1, 3)
    transformed = direction @ rot_matrix.T
    lon = np.arctan2(transformed[:, 0], transformed[:, 2])
    lat = np.arcsin(transformed[:, 1])
    map_x = (lon / (2 * np.pi) + 0.5) * w
    map_y = (lat / np.pi + 0.5) * h
    map_x = map_x.reshape(face_size, face_size).astype(np.float32)
    map_y = map_y.reshape(face_size, face_size).astype(np.float32)
    return map_x, map_y

def detect_and_compute(cube_faces):
    """
    Detect and compute features on cube faces.
    """
    keypoints = {}
    descriptors = {}
    for face in cube_faces:
        kp, des =  detectAndComputeKPandDescriptors(cube_faces[face])
        keypoints[face] = kp
        descriptors[face] = des
    return keypoints, descriptors

def match_features(descriptors1, descriptors2):
    """
    Match features between two sets of descriptors.
    """

    matches = {}
    for face in descriptors1:
        if descriptors1[face] is not None and descriptors2[face] is not None:
            matches[face] = getMatches(descriptors1[face], descriptors2[face])
        else:
            matches[face] = []
    return matches

def collect_matched_points(kp1, kp2, matches):
    """
    Collect matched points from all cube faces.
    """
    pts1 = []
    pts2 = []
    for face in matches:
        for m in matches[face]:
            pt1 = kp1[face][m.queryIdx].pt
            pt2 = kp2[face][m.trainIdx].pt
            # Adjust coordinates to account for face position
            x_offset = 0  # Modify if needed based on face arrangement
            y_offset = 0
            pts1.append((pt1[0] + x_offset, pt1[1] + y_offset))
            pts2.append((pt2[0] + x_offset, pt2[1] + y_offset))
    return np.array(pts1), np.array(pts2)

def estimate_pose(pts1, pts2, focal_length, pp):
    """
    Estimate the essential matrix and recover the pose.
    
    Parameters:
        pts1 (np.ndarray): Points from image 1.
        pts2 (np.ndarray): Corresponding points from image 2.
        focal_length (float): Approximate focal length.
        pp (tuple): Principal point (cx, cy).
    
    Returns:
        R (np.ndarray): Rotation matrix.
        t (np.ndarray): Translation vector.
    """
    # Camera Intrinsics Matrix
    K = np.array([[focal_length, 0, pp[0]],
              [0, focal_length, pp[1]],
              [0, 0, 1]])
    E, mask = cv2.findEssentialMat(pts1, pts2, focal=focal_length, pp=pp, method=cv2.RANSAC, threshold=1.0, prob=0.999)
    # Select inlier points
    inliers1 = pts1[mask.ravel() == 1]
    inliers2 = pts2[mask.ravel() == 1]
    _, R, t, mask_pose = cv2.recoverPose(E, inliers1, inliers2, focal=focal_length, pp=pp)
    
    return R, t, inliers1, inliers2

def refine_pose(R_initial, t_initial, points3D, pts2, K,verbose=False):
    # Convert rotation matrix to rotation vector
    rvec_initial, _ = cv2.Rodrigues(R_initial)
    tvec_initial = t_initial.ravel()
    params_initial = np.hstack((rvec_initial.ravel(), tvec_initial))

    def reprojection_error(params, points3D, pts2, K):
        rvec = params[:3].reshape(3, 1)
        tvec = params[3:6].reshape(3, 1)
        # Project 3D points into the second camera
        proj_pts, _ = cv2.projectPoints(points3D, rvec, tvec, K, np.zeros(5))
        proj_pts = proj_pts.reshape(-1, 2)
        error = pts2 - proj_pts
        return error.ravel()

    result = least_squares(
        reprojection_error,
        params_initial,
        verbose=verbose*2,
        ftol=1e-8,
        method='trf',
        loss='soft_l1',  # Use a robust loss function
        args=(points3D, pts2, K)
    )

    rvec_refined = result.x[:3].reshape(3, 1)
    tvec_refined = result.x[3:6].reshape(3, 1)
    R_refined, _ = cv2.Rodrigues(rvec_refined)
    t_refined = tvec_refined

    return R_refined, t_refined

def compute_R_t(imgLeft:cv2.typing.MatLike,imgRight:cv2.typing.MatLike, use_only_front=False, verbose=False, use_refined=True)->Tuple[np.array,np.array]:
    # Convert to cube maps
    cube_facesLeft = equirectangular_to_cubemap(imgLeft)
    cube_facesRight = equirectangular_to_cubemap(imgRight)
    if use_only_front:
        cube_facesLeft={"front":cube_facesLeft['front']} 
        cube_facesRight={"front":cube_facesRight['front']} 

    if verbose:
        for face in cube_facesLeft:  
            cv2.imshow(face, cube_facesLeft[face])
        cv2.waitKey(0)

    # Detect and compute features
    kp1, des1 = detect_and_compute(cube_facesLeft)
    kp2, des2 = detect_and_compute(cube_facesRight)

    # Match features
    matches = match_features(des1, des2)

    # Collect matched points
    pts1, pts2 = collect_matched_points(kp1, kp2, matches)

    # Approximate focal length and principal point
    h, w = cube_facesLeft['front'].shape[:2]
    focal_length = w / (2 * np.tan(np.deg2rad(90) / 2))  # Approximate for 90 degrees FOV
    pp = (w / 2, h / 2)

    # Estimate pose and triangulate 3D points
    R, t, inliers1, inliers2 = estimate_pose(pts1, pts2, focal_length, pp)
    rotation_angle = np.degrees(cv2.Rodrigues(R)[0].ravel())
    if not use_refined: 
        if verbose:
            print("Rotation (degrees):", rotation_angle)
            print("Translation Vector:\n", t.ravel())
        return R,t
    # Camera Intrinsics Matrix
    K = np.array([[focal_length, 0, pp[0]],
                [0, focal_length, pp[1]],
                [0, 0, 1]])

    inliers1_norm = cv2.undistortPoints(inliers1.reshape(-1, 1, 2), K, None)
    inliers2_norm = cv2.undistortPoints(inliers2.reshape(-1, 1, 2), K, None)
    P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = np.hstack((R, t))
    points4D_hom = cv2.triangulatePoints(P1, P2, inliers1_norm, inliers2_norm)
    points3D = (points4D_hom[:3] / points4D_hom[3]).T  # Shape: (N, 3)

    # Filter out invalid 3D points
    valid_idx = np.isfinite(points3D).all(axis=1)
    points3D = points3D[valid_idx]
    inliers2 = inliers2[valid_idx]

    # Keep points with positive depth
    valid_depth = points3D[:, 2] > 0
    points3D = points3D[valid_depth]
    inliers2 = inliers2[valid_depth]
    # Refine pose
    R_refined, t_refined = refine_pose(R, t, points3D, inliers2, K)


    # known_tx= -1.12
    # t*=known_tx/t[0]
    # t_refined*=known_tx/t_refined[0]

    # Rotation difference in degrees
    

    
    rotation_angle_refined = np.degrees(cv2.Rodrigues(R_refined)[0].ravel())

    if verbose:
        print("Rotation (degrees):", rotation_angle)
        print("Translation Vector:\n", t.ravel())

        # Output refined rotation and translation
        print("Refined Rotation Matrix:\n", R_refined)
        print("Refined Translation Vector:\n", t_refined.ravel())
        print("Refined Rotation (degrees):", rotation_angle_refined)
    

    return cv2.Rodrigues(R_refined)[0].ravel(), t_refined.ravel()

if __name__=="__main__":
    imgLeft_name =r"C:\Users\mmerl\projects\stereo_cam\Photos\test\image_R_21_20240730_143124_315000_2014.jpg"
    imgRight_name= r"C:\Users\mmerl\projects\stereo_cam\Photos\test\image_L_21_20240730_143131_519000_2301.jpg" 

    imgRight = cv2.imread(imgLeft_name)
    imgLeft = cv2.imread(imgRight_name)
    R,t=compute_R_t(imgRight,imgLeft)
    print(R,t)

    R,t=compute_R_t(imgLeft,imgRight, use_only_front=True)
    print(R,t)

