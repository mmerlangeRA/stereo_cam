import cv2
import numpy as np
import numpy.typing as npt
from typing import Tuple
from src.road_detection.common import AttentionWindow
from src.utils.coordinate_transforms import cartesian_to_spherical_array, spherical_to_equirectangular_array
from src.utils.image_processing import project_image_to_plane

class EquirectangularMapper:

    equirect_width:int
    equirect_height:int
    equirect_image:npt.NDArray[np.uint8]

    def __init__(self, equirect_width:int, equirect_height:int):
        self.equirect_width = equirect_width
        self.equirect_height = equirect_height
        self.equirect_image = np.zeros((self.equirect_height, self.equirect_width, 3), dtype=np.uint8)

    def map_3d_points_to_equirectangular(self,
            points_3d:npt.NDArray[np.float_],
            )->Tuple[npt.NDArray[np.int32], npt.NDArray[np.int32]]:
        """
        Maps 3D points onto an equirectangular image based on the plane's position and orientation.

        Parameters:
        - points_3d: The 3D points as a NumPy array of shape (N, 3).
        - plane_center, plane_normal, plane_up_vector, plane_width, plane_height: Parameters defining the plane.
        - equirect_width, equirect_height: Dimensions of the equirectangular image.

        Returns:
        - u,v array.
        """
        #set all values of equirect_image to 0
        self.equirect_image.fill(0)

        # Convert 3D points to spherical coordinates
        _, theta, phi = cartesian_to_spherical_array(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2])

        # Map to equirectangular coordinates
        u, v = spherical_to_equirectangular_array(theta, phi, self.equirect_width, self.equirect_height)

        # Flatten arrays for easier indexing
        u_flat = u.flatten()
        v_flat = v.flatten()

        # Round to nearest integer pixel coordinates
        u_int = np.floor(u_flat).astype(int)
        v_int = np.floor(v_flat).astype(int)

        # Wrap the u_int values to handle longitude wrapping (modulo operation)
        u_int = u_int % self.equirect_width  # Ensures u_int is within [0, equirect_width - 1]
        v_int = v_int % self.equirect_height

        return [u_int, v_int]

    def map_image_to_equirectangular(self,
            image_2d:npt.NDArray[np.uint8], 
            plane_center:npt.NDArray[np.float_], 
            plane_normal:npt.NDArray[np.float_],
            plane_up_vector:npt.NDArray[np.float_],
            plane_width:float, plane_height:float
            )->npt.NDArray[np.uint8]:
        """
        Maps a 2D image onto an equirectangular image based on the plane's position and orientation.
        
        Parameters:
        - image_2d: The 2D image as a NumPy array of shape (H, W, C).
        - plane_center, plane_normal, plane_up_vector, plane_width, plane_height: Parameters defining the plane.
        - equirect_width, equirect_height: Dimensions of the equirectangular image.
        
        Returns:
        - equirect_image: The equirectangular image with the 2D image projected onto it.
        """
        #set all values of equirect_image to 0
        self.equirect_image.fill(0)
        
        # Compute 3D points on the plane
        points_3d = project_image_to_plane(
            image_2d, plane_center, plane_normal, plane_up_vector, plane_width, plane_height)
        x = points_3d[:, :, 0]
        y = points_3d[:, :, 1]
        z = points_3d[:, :, 2]
        
        # Convert to spherical coordinates
        _, theta, phi = cartesian_to_spherical_array(x, y, z)
        
        # Map to equirectangular coordinates
        u, v = spherical_to_equirectangular_array(theta, phi, self.equirect_width, self.equirect_height)
        
        # Flatten arrays for easier indexing
        u_flat = u.flatten()
        v_flat = v.flatten()
        pixels = image_2d.reshape(-1, 3)

        # Round to nearest integer pixel coordinates
        u_int = np.floor(u_flat).astype(int)
        v_int = np.floor(v_flat).astype(int)

        # Wrap the u_int values to handle longitude wrapping (modulo operation)
        u_int = u_int % self.equirect_width  # Ensures u_int is within [0, equirect_width - 1]
        v_int = v_int % self.equirect_height
        # Clamp the v_int values to the valid range [0, equirect_height - 1]
        #v_int = np.clip(v_int, 0, equirect_height - 1)

        # Now, no need for valid_mask since u_int and v_int are within valid ranges
        # Assign pixel values to equirectangular image
        self.equirect_image[v_int, u_int] = pixels
        
        return self.equirect_image
