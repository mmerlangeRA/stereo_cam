from typing import Tuple
import cv2
import numpy as np
import numpy.typing as npt


from src.road_detection.common import AttentionWindow
from src.utils.coordinate_transforms import cartesian_to_equirectangular, cartesian_to_spherical, pixel_to_spherical, spherical_to_cartesian
from src.matching.match_simple_pytorch import VGGMatcher
from src.utils.equirectangular.equirectangular_mapper import EquirectangularMapper
from src.utils.equirectangular.minimize_projection import  optimize_roadsign_position_and_orientation
from src.utils.equirectangular.SignTools import SignMatcherTool
from src.utils.TransformClass import Transform
from src.utils.path_utils import get_data_path

class SignMatcher:

    matcher:VGGMatcher
    estimated_sign_height = 0.7
    equirectangularMapper: EquirectangularMapper = None

    def __init__(self,reference_folder_path:str=get_data_path('panneaux') ) -> None:
        self.matcher = VGGMatcher(reference_folder_path)

    def get_equiRectangularMapper(self,equi_width:int,equi_height:int )->EquirectangularMapper:
        if self.equirectangularMapper is None or equi_height != self.equirectangularMapper.equirect_height or equi_width != self.equirectangularMapper.equirect_width:
            self.equirectangularMapper= EquirectangularMapper(equi_width,equi_height)
        return self.equirectangularMapper

    def find_matching_sign_id(self, equirect_image:npt.NDArray[np.uint8], signWindow:AttentionWindow, debug=False)->Tuple[int,float]:
        query_image = signWindow.crop_image(equirect_image)
        id,score= self.matcher.find_matching(query_image)
        if debug:
            cv2.imshow("query", query_image)
            cv2.waitKey(0)
        return id,score

    def crop_all_images(self):
        self.matcher.crop_all_images()

    def get_image_path_from_id(self, id:int)->str:
        return self.matcher.get_image_path(id)
    
    def get_signImage_from_id(self, id:int)->npt.NDArray[np.uint8]:
        return cv2.imread(self.matcher.get_image_path(id))
    
    def estimate_initial_sign_transform(self, equirect_image:npt.NDArray[np.uint8],signWindow:AttentionWindow, signImage:npt.NDArray[np.uint8] ,debug=False)->Transform:

        signMatcherTool=SignMatcherTool()
        
        sign_center = signWindow.center
        sign_top_center = signWindow.top_center
        sign_bottom_center = signWindow.bottom_center

        
        h,w= signImage.shape[:2]
        plane_height = self.estimated_sign_height  # Adjust as needed
        plane_width = plane_height*w/h

        equirect_height,equirect_width = equirect_image.shape[:2]

        theta,phi = pixel_to_spherical(equirect_width, equirect_height,sign_top_center[0],sign_top_center[1])
        x_t,y_t,z_t = spherical_to_cartesian(theta,phi)

        theta,phi = pixel_to_spherical(equirect_width, equirect_height,sign_bottom_center[0],sign_bottom_center[1])
        x_b,y_b,z_b = spherical_to_cartesian(theta,phi)



        # theta,phi = pixel_to_spherical(equirect_width, equirect_height,sign_center[0],sign_center[1])
        # x_c,y_c,z_c = spherical_to_cartesian(theta,phi)

        #Let's guess initial position
        #Assume Sign is at 5m, vertical
        #x_c, y_c, z_c = signMatcherTool.normalize_to_z1(x_c, y_c, z_c)

        
        x_t, y_t, z_t = signMatcherTool.normalize_to_z1(x_t, y_t, z_t,5.)
        x_b, y_b, z_b = signMatcherTool.normalize_to_z1(x_b, y_b, z_b,5.)

        # Compute the vertical distance between top and bottom in normalized coordinates
        height_in_y = y_b - y_t

        # Handle potential division by zero
        if height_in_y == 0:
            raise ValueError("The vertical distance between the top and bottom is zero.")

        # Compute the scaling ratio based on the estimated sign height
        
        computed_height = np.linalg.norm(np.array([x_b,y_b, z_b])-np.array([x_t,y_t, z_t]))

        ratio = self.estimated_sign_height / computed_height*z_t
        x_t, y_t, z_t = signMatcherTool.normalize_to_z1(x_t, y_t, z_t,ratio)
        x_b, y_b, z_b = signMatcherTool.normalize_to_z1(x_b, y_b, z_b,ratio)
        # Scale the positions accordingly
        # x_c, y_c, z_c = signMatcherTool.scale_positions(x_c, y_c, z_c, ratio)
        # x_t, y_t, z_t = signMatcherTool.scale_positions(x_t, y_t, z_t, ratio)
        # x_b, y_b, z_b = signMatcherTool.scale_positions(x_b, y_b, z_b, ratio)



        x_c = (x_t+x_b)/2.
        y_c = (y_t + y_b)/2.
        z_c = (z_t + z_b)/2

        #y_c+=self.estimated_sign_height/2.
        u_t,v_t = cartesian_to_equirectangular(x_t,y_t,z_t,equirect_width,equirect_height)
        print(sign_top_center, u_t,v_t)
        u_b,v_b = cartesian_to_equirectangular(x_b,y_b,z_b,equirect_width,equirect_height)
        print(sign_bottom_center, u_b,v_b)
        u_c,v_c = cartesian_to_equirectangular(x_c,y_c,z_c,equirect_width,equirect_height)
        print([x_c,y_c,z_c], u_c,v_c)

        yaw_estimate = pitch_estimate = 0.
        roll_estimate= 0.
        roll_estimate = 0.

        estimated_sign_transform = Transform(x_c, y_c, z_c, yaw_estimate, pitch_estimate, roll_estimate)

        if debug:
            print(f"estimated_sign_transform {estimated_sign_transform}")

        return estimated_sign_transform
    
    def optimize_sign_position_and_orientation(self, equirect_image:npt.NDArray[np.uint8],sign_img: npt.NDArray[np.uint8], signWindow:AttentionWindow, estimatedTransform:Transform)->Transform:
        h,w=sign_img.shape[:2]
        plane_height = self.estimated_sign_height
        plane_width = plane_height*w/h
        equirect_height, equirect_width = equirect_image.shape[:2]
        optimizedTransform = optimize_roadsign_position_and_orientation(
            estimatedTransform,
            equirect_image,
            sign_img,
            signWindow,
            plane_width,
            plane_height,
            equirect_width,
            equirect_height
        )

        return optimizedTransform
    
    def get_top_bottom_projected(self,signImage:npt.NDArray[np.uint8],signTransform:Transform, equi_width:int, equi_height:int):
        equirectangularMapper = self.get_equiRectangularMapper(equi_width,equi_height)
        plane_height = self.estimated_sign_height
        h,w = signImage.shape[:2]

        optimized_plane_center = np.array([signTransform.xc, signTransform.yc, signTransform.zc], dtype=np.float64)
        # Recompute rotation matrix and orientation vectors
        rotation_matrix = signTransform.rotationMatrix

        optimized_plane_up_vector = rotation_matrix @ np.array([0, 1, 0], dtype=np.float64)
        plane_up_vector_rotated = rotation_matrix @ optimized_plane_up_vector
        top_sign = optimized_plane_center - plane_height * 0.5 * plane_up_vector_rotated
        bottom_sign = optimized_plane_center + plane_height * 0.5 * plane_up_vector_rotated

        # Convert points_3D to a NumPy array
        points_3D = np.array([top_sign, bottom_sign])  # Shape: (2, 3)

        us,vs = equirectangularMapper.map_3d_points_to_equirectangular(points_3D)
        return us,vs
    
    def map_to_equirectangular(self, signTransform:Transform,signImage:npt.NDArray[np.uint8],equi_width:int,equi_height:int,estimated_sign_height:float=-1)->np.ndarray:
        equirectangularMapper = self.get_equiRectangularMapper(equi_width,equi_height)
        
        if estimated_sign_height > 0.:
            self.estimate_initial_sign_transform = estimated_sign_height

        center_x,center_y,center_z,yaw,pitch,roll = signTransform.as_array()

        h,w= signImage.shape[:2]
        plane_height = self.estimated_sign_height  # Adjust as needed
        plane_width = plane_height*w/h

        plane_center = np.array([center_x, center_y, center_z], dtype=np.float64)

        # Recompute rotation matrix and orientation vectors
        
        equirect_sign_image = equirectangularMapper.map_image_to_equirectangular(
            signImage, plane_center, yaw,pitch,roll, plane_width, plane_height)
        return equirect_sign_image
    
