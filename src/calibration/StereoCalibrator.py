from dataclasses import asdict, dataclass, field
import json
import os
from typing import List, Tuple
import cv2
import numpy as np
import yaml
from src.features_2d.utils import detectAndComputeKPandDescriptors
from scipy.optimize import least_squares

from src.calibration.cube import compute_cube_calibration, undistort_and_crop
from src.calibration.stereo_standard_refinement import compute_auto_calibration_for_2_stereo_standard_images
from src.utils.cube_image import get_cube_front_image
from src.utils.path_utils import get_calibration_folder_path

@dataclass
class StereoFullCalibration:
    mono_img_width:int = -1
    mono_img_height:int = -1
    mono_K:np.ndarray = field(default_factory=lambda: np.array([]))
    mono_dist: cv2.typing.MatLike = field(default_factory=lambda: np.array([]))
    mono_ret: float =-1.
    
    stereo_undistorted_img_width:int = -1
    stereo_undistorted_img_height:int = -1
    stereo_undistorted_K:np.ndarray = field(default_factory=lambda: np.array([]))
    stereo_undistorted_rvec:np.ndarray = field(default_factory=lambda: np.array([]))
    stereo_undistorted_tvec:np.ndarray = field(default_factory=lambda: np.array([]))
    stereo_undistorted_cost:float = -1.

    stereo_rectified_img_width:int = -1
    stereo_rectified_img_height:int = -1
    stereo_rectified_K:np.ndarray = field(default_factory=lambda: np.array([]))
    stereo_rectified_rvec:np.ndarray = field(default_factory=lambda: np.array([]))
    stereo_rectified_tvec:np.ndarray = field(default_factory=lambda: np.array([]))
    stereo_rectified_cost:float = -1.
    stereo_rectified_Z0: float = -1.

    def to_json(self) -> str:
        # Convert the dataclass to a dictionary
        dict_representation = asdict(self)

        # Convert any numpy arrays to lists
        for key, value in dict_representation.items():
            if isinstance(value, np.ndarray):
                dict_representation[key] = value.tolist()

        return json.dumps(dict_representation)

    def from_json(json_str):
        d = json.loads(json_str)
        return StereoFullCalibration(**d)

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
            [0, fx, cy],
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

class StereoCalibrator:
    verbose: bool
    calibration: StereoFullCalibration

    calibration_file_name : str
    calibration_file_path :str
    estimated_base_line_in_m:float #along x


    def __init__(self,estimated_base_line_in_m=1.12,calibration_file_name="calibration_matrix.json",verbose=False) -> None:
        self.estimated_base_line_in_m = estimated_base_line_in_m
        self.calibration_file_name = calibration_file_name
        self.calibration_file_path = get_calibration_folder_path(self.calibration_file_name)
        self.verbose = verbose
        self.calibration = StereoFullCalibration()

    def reset(self)->None:
        self.calibration_file_path = get_calibration_folder_path(self.calibration_file_name)
        if os.path.exists(self.calibration_file_path)==True:    
            os.remove(self.calibration_file_path)
        
    def save_calibration(self)->None:
        # and save it to a file
        open(self.calibration_file_path, 'w').write(self.calibration.to_json())

    def read_calibration(self)->None:
        if os.path.exists(self.calibration_file_path)==False: 
            raise FileNotFoundError(f"Calibration file not found at {self.calibration_file_path}")

        self.calibration = StereoFullCalibration.from_json (open(self.calibration_path, 'r').read())
 

    def compute_mono_chessboard_calibration(self, image_paths:List[str],chessboard_size:cv2.typing.Size,square_size:float)->tuple[float,cv2.typing.MatLike, cv2.typing.MatLike]:
        self.calibration.mono_K, self.calibration.mono_dist,self.calibration.mono_ret = compute_cube_calibration(image_paths=image_paths, chessboard_size=chessboard_size, square_size=square_size, verbose=self.verbose)
        img = cv2.imread(image_paths[0])
        front_image=get_cube_front_image(img)
        height, width = front_image.shape[:2]
        self.calibration.mono_img_width=width
        self.calibration.mono_img_height=height
        return self.calibration.mono_K, self.calibration.mono_dist,self.calibration.mono_ret
    
    def undistort_and_crop(self,leftImg)-> tuple[cv2.typing.MatLike,cv2.typing.MatLike]:
        undistorted_left,newcameramtx = undistort_and_crop(leftImg, self.calibration.mono_K, self.calibration.mono_dist)
        return undistorted_left,newcameramtx
    
    def compute_global_auto_calibration(self,image_paths_left: List[str], 
                                        image_paths_right: List[str]) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        """
        Automatically calibrates a stereo camera setup using multiple pairs of standard images.
        CamLeft is the world referential.

        Args:
            image_paths_left (List[str]): List of paths to the left stereo images.
            image_paths_right (List[str]): List of paths to the right stereo images.
            verbose (bool): If True, prints and saves intermediate results. Default is False.

        Returns:
            Tuple[np.ndarray, float, np.ndarray, np.ndarray]: 
                - K: Refined camera matrix.
                - cost: Final reprojection error cost.
                - refined_rvec: Refined rotation vector.
                - refined_tvec: Refined translation vector.
        """
        if len(image_paths_left) != len(image_paths_right):
            raise ValueError("The number of left and right images must be the same.")

        # Initialize lists to accumulate all keypoints and matches
        all_pts1 = []
        all_pts2 = []
        
        for i in range(len(image_paths_left)):
            # Read images
            imgLeft = cv2.imread(image_paths_left[i], cv2.IMREAD_GRAYSCALE)
            imgRight = cv2.imread(image_paths_right[i], cv2.IMREAD_GRAYSCALE)
            
            if imgLeft is None or imgRight is None:
                raise ValueError(f"Could not read one of the images: {image_paths_left[i]} or {image_paths_right[i]}")

            # Detect and compute features for both images
            keypoints_list, descriptors_list = zip(*[detectAndComputeKPandDescriptors(img) for img in [imgLeft, imgRight]])

            if any(d is None for d in descriptors_list):
                raise ValueError(f"Could not find enough descriptors in one or both images: {image_paths_left[i]}, {image_paths_right[i]}")
            
            # Create BFMatcher object
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

            # Match descriptors between the two images
            matches = bf.match(descriptors_list[0], descriptors_list[1])

            if len(matches) < 8:
                raise ValueError(f"Not enough matches found between images: {image_paths_left[i]}, {image_paths_right[i]}")

            matches = sorted(matches, key=lambda x: x.distance)

            if self.verbose:
                # Draw matches
                img_matches = cv2.drawMatches(imgLeft, keypoints_list[0], imgRight, keypoints_list[1], matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                cv2.imwrite(f'img_matches_{i}.png', img_matches)

            # Extract matched keypoints
            pts1 = np.float32([keypoints_list[0][m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            pts2 = np.float32([keypoints_list[1][m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            y_threshold = 30.0  # Adjust this threshold as needed

            # Filter out points with y-position differences greater than the threshold
            pts1, pts2 = zip(*[(p1, p2) for p1, p2 in zip(pts1, pts2) if abs(p1[0][1] - p2[0][1]) <= y_threshold])

            # Convert lists back to numpy arrays
            pts1 = np.array(pts1).reshape(-1, 1, 2)
            pts2 = np.array(pts2).reshape(-1, 1, 2)

            if len(pts1) < 8:
                raise ValueError(f"Not enough valid matches after filtering by y-position differences for images: {image_paths_left[i]}, {image_paths_right[i]}")

            # Accumulate points
            all_pts1.append(pts1)
            all_pts2.append(pts2)

            # Stack all points for global optimization
            all_pts1 = np.vstack(all_pts1)
            all_pts2 = np.vstack(all_pts2)

            h, w = imgLeft.shape[:2]

            # Initial guess 
            fx_init = fy_init = 700.0   # in pixels
            cx_init = w / 2
            cy_init = h / 2
            rvec_init = np.array([[-0.0], [0.0], [-0.0]], dtype=np.float32)
            tvec_init = np.array([[-self.estimated_base_line_in_m], [0.0], [0.0]], dtype=np.float32)
            tvec_init = np.array([[self.estimated_base_line_in_m], [0.0], [0.0]], dtype=np.float32)

            # Initial parameter guess (focal length and principal point)
            initial_params = np.hstack(([fx_init], [fy_init], [cx_init], [cy_init], rvec_init.ravel(), tvec_init.ravel()))
            bounds = (
                [100, 100, w / 2.5, h / 2.5, -0.1, -0.1, -0.1, -1.13, -0.03, -0.03],
                [2000, 2000, w / 1.5, h / 1.5, 0.1, 0.1, 0.1, -1.11, 0.03, 0.03]
            )

            # Perform bundle adjustment on all points
            result = least_squares(compute_reprojection_residual, initial_params, args=(all_pts1, all_pts2, np.zeros(5)), bounds=bounds,loss='huber')
            
            # Extract refined parameters
            refined_params = result.x
            refined_fx, refined_fy = refined_params[0], refined_params[1]
            refined_cx, refined_cy = refined_params[2], refined_params[3]
            refined_rvec = refined_params[4:7].reshape(3, 1)
            refined_tvec = refined_params[7:10].reshape(3, 1)
            self.estimated_base_line_in_m = refined_tvec[0][0]

            # Update the camera matrix with the refined focal length and principal point
            K = np.array([[refined_fx, 0, refined_cx],
                        [0, refined_fy, refined_cy],
                        [0, 0, 1]])

            if self.verbose:
                print(f"Refined Focal Length (fx, fy): ({refined_fx:.2f}, {refined_fy:.2f})")
                print(f"Refined Principal Point (cx, cy): ({refined_cx:.2f}, {refined_cy:.2f})")
                print(f"Refined Rotation Vector:\n{refined_rvec}")
                print(f"Refined Translation Vector:\n{refined_tvec}")
                print(f"Updated Camera Matrix:\n{K}")
                print(f"Final Reprojection Error Cost: {result.cost:.6f}")

            
            return K, float(result.cost), refined_rvec, refined_tvec
    
    def compute_global_auto_calibration_undistorted(self,image_paths_left: List[str], 
                                        image_paths_right: List[str]) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        imgLeft = cv2.imread(image_paths_left[0])
        self.calibration.stereo_undistorted_img_height, self.calibration.stereo_undistorted_img_width = imgLeft.shape[:2]
        K, cost, refined_rvec, refined_tvec = self.compute_global_auto_calibration(image_paths_left, image_paths_right)
        self.calibration.stereo_undistorted_K=K
        self.calibration.stereo_undistorted_rvec=refined_rvec
        self.calibration.stereo_undistorted_tvec=refined_tvec
        self.calibration.stereo_undistorted_cost=cost
        return  K, cost, refined_rvec, refined_tvec
    
    def compute_global_auto_calibration_rectified(self,image_paths_left: List[str], 
                                        image_paths_right: List[str]) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        imgLeft = cv2.imread(image_paths_left[0])
        self.calibration.stereo_rectified_img_height, self.calibration.stereo_rectified_img_width = imgLeft.shape[:2]
        K, cost, refined_rvec, refined_tvec = self.compute_global_auto_calibration(image_paths_left, image_paths_right)
        self.calibration.stereo_rectified_K=K
        self.calibration.stereo_rectified_rvec=refined_rvec
        self.calibration.stereo_rectified_tvec=refined_tvec
        self.calibration.stereo_rectified_cost=cost
        self.compute_stereo_rectified_Z0()
        return  K, cost, refined_rvec, refined_tvec
        
    def compute_stereo_rectified_Z0(self):
        half_rot_y = self.calibration.stereo_rectified_rvec[1][0]/2.
        half_baseline = self.calibration.stereo_rectified_tvec[0][0]/2.
        self.calibration.stereo_rectified_Z0 = half_baseline * np.tan(np.pi/2.-half_rot_y)
        print(type(self.calibration.stereo_rectified_Z0))
    
    def compute_auto_calibration_for_2_stereo_standard_images(self,imgLeft:cv2.typing.MatLike, imgRight:cv2.typing.MatLike,verbose=True)-> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
       
       return compute_auto_calibration_for_2_stereo_standard_images(imgLeft, imgRight, verbose)
  
    def rectify_undistorted_images(self,undistorted_left:cv2.typing.MatLike,undistorted_right:cv2.typing.MatLike)-> tuple[cv2.typing.MatLike,cv2.typing.MatLike]:
        """
        Rectify already undistorted images
        """
        if self.calibration.stereo_undistorted_K is None or len(self.calibration.stereo_undistorted_K) ==0 :
            self.calibration.stereo_undistorted_K,_,self.calibration.stereo_undistorted_rvec,self.calibration.stereo_undistorted_tvec =self.compute_auto_calibration_for_2_stereo_standard_images(undistorted_left,undistorted_right)
        
        K = self.calibration.stereo_undistorted_K

        A1 = K # Left camera matrix intrinsic
        A2 = K # Right camera matrix intrinsic
        
        RT1 = np.eye(3, 4)

        rvec=self.calibration.stereo_undistorted_rvec
        tvec=self.calibration.stereo_undistorted_tvec

        # Convert the rotation vector to a rotation matrix
        R, _ = cv2.Rodrigues(rvec)
        
        # Create the 3x4 extrinsic matrix by combining R and tvec
        RT2 = np.hstack((R, tvec))
        
        # Original projection matrices
        Po1 = A1.dot( RT1 )
        Po2 = A2.dot( RT2 )

        # Camera centers (world coord.)
        C1 = -np.linalg.inv(Po1[:,:3]).dot(Po1[:,3])
        C2 = -np.linalg.inv(Po2[:,:3]).dot(Po2[:,3])

        # Transformations
        T1to2 = C2 - C1 # Translation from first to second camera
        R1to2 = RT2[:,:3].dot(np.linalg.inv(RT1[:,:3])) # Rotation from first to second camera (3x3)

        R1, R2, Pn1, Pn2, _, _, _ = cv2.stereoRectify(A1, np.zeros((1,5)), A2, np.zeros((1,5)), (undistorted_left.shape[1], undistorted_right.shape[0]), R1to2, T1to2, alpha=-1 )

        # Rectify1 = R1.dot(np.linalg.inv(A1))
        # Rectify2 = R2.dot(np.linalg.inv(A2))

        mapL1, mapL2 = cv2.initUndistortRectifyMap(A1, np.zeros((1,5)), R1, Pn1, (undistorted_left.shape[1], undistorted_left.shape[0]), cv2.CV_32FC1)
        mapR1, mapR2 = cv2.initUndistortRectifyMap(A2, np.zeros((1,5)), R2, Pn2, (undistorted_right.shape[1], undistorted_right.shape[0]), cv2.CV_32FC1)

        img1_rect = cv2.remap(undistorted_left, mapL1, mapL2, cv2.INTER_LINEAR)
        img2_rect = cv2.remap(undistorted_right, mapR1, mapR2, cv2.INTER_LINEAR)
        return img1_rect, img2_rect



