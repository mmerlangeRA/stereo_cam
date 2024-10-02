from typing import Tuple, Union
import numpy as np
import numpy.typing as npt
import cv2

import numpy as np
from typing import Union, Tuple

def pixel_to_spherical(image_width: int, image_height: int, pixel_x: Union[int, np.ndarray], pixel_y: Union[int, np.ndarray]) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    """
    Convert pixel coordinates to spherical coordinates (theta, phi).
    
    This function handles both single pixel coordinates and arrays of pixel coordinates.

    Parameters:
    - image_width: Width of the equirectangular image.
    - image_height: Height of the equirectangular image.
    - pixel_x: Int or NumPy array of x pixel coordinates.
    - pixel_y: Int or NumPy array of y pixel coordinates.

    Returns:
    - theta: Float or NumPy array of longitude angles (theta).
    - phi: Float or NumPy array of latitude angles (phi).
    """
    # Handle scalar inputs
    if np.isscalar(pixel_x) and np.isscalar(pixel_y):
        theta = (pixel_x / image_width) * 2 * np.pi - np.pi  # Longitude
        phi = (pixel_y / image_height) * np.pi - np.pi / 2   # Latitude
        return theta, phi

    # Ensure pixel_x and pixel_y are NumPy arrays for vectorized operations
    pixel_x = np.asarray(pixel_x)
    pixel_y = np.asarray(pixel_y)

    # Compute theta (longitude) and phi (latitude) for arrays of pixel coordinates
    theta = (pixel_x / image_width) * 2 * np.pi - np.pi  # Longitude
    phi = (pixel_y / image_height) * np.pi - np.pi / 2   # Latitude

    return theta, phi

def spherical_to_cartesian(theta: Union[float, np.ndarray], phi: Union[float, np.ndarray]) -> np.ndarray:
    """
    Convert spherical coordinates to 3D cartesian coordinates.
    
    This function handles both single float inputs and arrays of inputs.

    Parameters:
    - theta: Float or NumPy array of longitude angles (theta).
    - phi: Float or NumPy array of latitude angles (phi).

    Returns:
    - A NumPy array containing the (x, y, z) cartesian coordinates.
      If the inputs are scalars, the output will be a 1D array.
      If the inputs are arrays, the output will be a 2D array of shape (N, 3), where N is the number of points.
    """
    # Handle scalar inputs
    if np.isscalar(theta) and np.isscalar(phi):
        x = np.cos(phi) * np.sin(theta)
        y = np.sin(phi)
        z = np.cos(phi) * np.cos(theta)
        return np.array([x, y, z])
    
    # Ensure theta and phi are NumPy arrays for vectorized operations
    theta = np.asarray(theta)
    phi = np.asarray(phi)

    # Compute x, y, z for each spherical coordinate
    x = np.cos(phi) * np.sin(theta)
    y = np.sin(phi)
    z = np.cos(phi) * np.cos(theta)

    # Stack x, y, z into a single array with shape (N, 3)
    cartesian_coords = np.stack((x, y, z), axis=-1)

    return cartesian_coords

def pixel_to_cartesian(image_width: int, image_height: int, pixel_x: Union[float, np.ndarray], pixel_y: Union[float, np.ndarray]) -> np.ndarray:
    """
    Convert pixel coordinates to 3D cartesian coordinates.

    Parameters:
    - image_width: Width of the image.
    - image_height: Height of the image.
    - pixel_x: x-coordinate of the pixel(s). Can be a single integer or a NumPy array.
    - pixel_y: y-coordinate of the pixel(s). Can be a single integer or a NumPy array.

    Returns:
    - Cartesian coordinates as NumPy array(s). If input is scalar, output is (3,) array. 
      If input is array, output is (N, 3) array.
    """
    # Check if input is a scalar or an array
    is_scalar = np.isscalar(pixel_x) and np.isscalar(pixel_y)
    
    # If inputs are scalars, make them arrays for consistent processing
    if is_scalar:
        pixel_x = np.array([pixel_x])
        pixel_y = np.array([pixel_y])

    # Compute theta and phi for either single pixel or array of pixels
    theta, phi = pixel_to_spherical(image_width, image_height, pixel_x, pixel_y)
    
    # Convert spherical coordinates to cartesian coordinates
    cartesian_coords = spherical_to_cartesian(theta, phi)
    
    # If inputs were scalars, return a (3,) array, otherwise return (N, 3) array
    return cartesian_coords[0] if is_scalar else cartesian_coords

def cartesian_to_spherical_old(x:float,y:float,z:float) ->  Tuple[float, float,float]:
    """Convert 3D cartesian coordinates to spherical coordinates ."""
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(x, z) # Azimuth angle
    phi = np.arcsin(y / r)    # Elevation angle
    return np.array([r,theta, phi])

def cartesian_to_spherical(x: Union[float, np.ndarray], y: Union[float, np.ndarray], z: Union[float, np.ndarray]) -> Union[Tuple[float, float, float], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Convert 3D cartesian coordinates to spherical coordinates.

    Parameters:
    - x: X coordinate (can be scalar or array).
    - y: Y coordinate (can be scalar or array).
    - z: Z coordinate (can be scalar or array).

    Returns:
    - Tuple of (r, theta, phi):
      - r: Radial distance (same shape as input).
      - theta: Azimuth angle (same shape as input).
      - phi: Elevation angle (same shape as input).
    """
    # Convert inputs to arrays if they aren't already
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    # Compute r, theta, phi
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(x, z)  # Azimuth angle
    phi = np.arcsin(y / r)    # Elevation angle

    # Return single value if inputs were scalars, else return arrays
    if r.size == 1:
        return float(r), float(theta), float(phi)
    else:
        return r, theta, phi


def spherical_to_equirectangular(theta:float, phi:float, image_width:int, image_height:int) -> Tuple[int, int]:
    """Convert spherical coordinates to equirectangular pixel coordinates."""
    u = (theta + np.pi) / (2 * np.pi) * image_width
    v = (phi + np.pi / 2) / np.pi * image_height
    return int(u), int(v)


def spherical_to_equirectangular(
    theta: Union[float, np.ndarray], 
    phi: Union[float, np.ndarray], 
    image_width: int, 
    image_height: int,
    to_int = True
) -> Union[Tuple[int, int], Tuple[np.ndarray, np.ndarray]]:
    """
    Convert spherical coordinates to equirectangular pixel coordinates.
    
    Parameters:
    - theta: Longitude angle(s) in radians (can be scalar or array).
    - phi: Latitude angle(s) in radians (can be scalar or array).
    - image_width: Width of the equirectangular image in pixels.
    - image_height: Height of the equirectangular image in pixels.

    Returns:
    - u, v: Equirectangular pixel coordinates (as int if scalar inputs, or arrays if input is array).
    """
    # Ensure theta and phi are arrays for vectorized operations
    theta = np.asarray(theta)
    phi = np.asarray(phi)

    # Compute equirectangular pixel coordinates
    u = (theta + np.pi) / (2 * np.pi) * image_width
    v = (phi + np.pi / 2) / np.pi * image_height

    if not to_int: return u, v
    # Convert to integers
    u_int = np.floor(u).astype(int)
    v_int = np.floor(v).astype(int)

    # If inputs were scalars, return as scalars
    if u_int.size == 1:
        return int(u_int), int(v_int)
    else:
        return u_int, v_int


def cartesian_to_equirectangular(x:Union[float, np.ndarray], y:Union[float, np.ndarray], z:Union[float, np.ndarray], image_width:int, image_height:int,to_int=True) -> Tuple[int, int]:
    """Convert 3D cartesian coordinates to equirectangular pixel coordinates."""
    _,theta, phi = cartesian_to_spherical(x, y, z)
    return spherical_to_equirectangular(theta, phi, image_width, image_height,to_int=to_int)

def get_transformation_matrix(rvec:np.array, tvec:np.array)->np.array:
    R, _ = cv2.Rodrigues(rvec)
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = R
    transformation_matrix[:3, 3] = tvec.flatten()
    return transformation_matrix

def get_rvec_tvec_from_transformation_matrix(transformation_matrix:np.array)->Tuple[np.array,np.array]:
    R = transformation_matrix[:3, :3]
    tvec = transformation_matrix[:3, 3].reshape(3, 1)
    rvec, _ = cv2.Rodrigues(R)
    return rvec, tvec

def invert_rvec_tvec(rvec:np.array, tvec:np.array)->Tuple[np.array,np.array]:
    transformation_matrix = get_transformation_matrix(rvec, tvec)
    inv_transformation_matrix = np.linalg.inv(transformation_matrix)
    rvec_inv,tvec_inv = get_rvec_tvec_from_transformation_matrix(inv_transformation_matrix)
    return rvec_inv, tvec_inv

def get_identity_extrinsic_matrix()->np.array:
    return np.eye(3, 4)

def get_extrinsic_matrix_from_rvec_tvec(rvec:np.array, tvec:np.array)->np.array:
    R, _ = cv2.Rodrigues(rvec)
    RT = np.hstack((R, tvec))
    return RT

def eac_to_road_plane(imgWidth:int, imgHeight:int,road_rvec:np.array,camHeight:float,contour_x,contour_y):
        # We assume world is the camera referential and we want projection on road plane
        # road_vec is transformation of the road in world space

        rotation_matrix, _ = cv2.Rodrigues(road_rvec)
        #inv_transformation_matrix = np.linalg.inv(transformation_matrix)
        #let's work in plane referential
        road_points=[]
        for i in range(len(contour_x)):
            x = contour_x[i]
            y= contour_y[i]
            plane_dy = -camHeight
            theta, phi = pixel_to_spherical (imgWidth, imgHeight,x, y)
            world_ray = spherical_to_cartesian(theta, phi)
            road_plane_ray = rotation_matrix  @ world_ray
            l = plane_dy/road_plane_ray[1]
            p = l*road_plane_ray
            road_points.append([p[0],p[2]])
        
        road_points = np.array(road_points)

        # Find the minimum x value
        min_x = np.min(road_points[:, 0])
        max_x = np.max(road_points[:, 0])
        min_z = np.min(road_points[:, 1])
        max_z = np.max(road_points[:, 1])

        width = int(max_x - min_x+1)
        height = int(max_z - min_z+1)
        
        road_points[:,0] = (road_points[:,0] - min_x)
        road_points[:,1] = (road_points[:,1] - min_z)
        return road_points,width, height

if __name__ == "__main__":
    # Example usage
    rvec = np.array([[0.1], [0.2], [0.3]], dtype=np.float32)
    tvec = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)

    rvec_inv, tvec_inv = invert_rvec_tvec(rvec, tvec)

    print("Inverted rvec:\n", rvec_inv)
    print("Inverted tvec:\n", tvec_inv)

    R, _ = cv2.Rodrigues(rvec)
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = R
    transformation_matrix[:3, 3] = tvec.flatten()

    R_inv, _ = cv2.Rodrigues(rvec_inv)
    transformation_matrix_inverse =np.eye(4)
    transformation_matrix_inverse[:3, :3] = R_inv
    transformation_matrix_inverse[:3, 3] = tvec_inv.flatten()

    m = transformation_matrix.dot(transformation_matrix_inverse)
    print(m)

    transformation_matrix = get_transformation_matrix(rvec, tvec)
    transformation_matrix_inverse = get_transformation_matrix(rvec_inv, tvec_inv)
    m = transformation_matrix.dot(transformation_matrix_inverse)
    print(m)


def intersect_ray_plane(plane_coeffs, ray_direction):
    # plane equation is ax+by+cz+d=0
    # Unpack the plane normal and ray direction
    a, b, c, d = plane_coeffs
    dx, dy, dz = ray_direction

    # Compute the denominator (dot product of plane normal and ray direction)
    denominator = a * dx + b * dy + c * dz

    # Check if the ray is parallel to the plane (denominator is zero)
    if np.isclose(denominator, 0):
        return None  # No intersection, the ray is parallel to the plane

    # Compute the value of t
    t = -d / denominator

    # Calculate the intersection point
    intersection_point = t * np.array(ray_direction)

    return intersection_point
    ''' Example usage:
    plane_params = np.array([1, -1, 2,2])  # Example normal vector (a, b, c)
    ray_direction = np.array([2, 3, 1])  # Example ray direction (dx, dy, dz)

    intersection = intersect_ray_plane(plane_params, ray_direction)

    if intersection is not None:
        print(f"Intersection point: {intersection}")
    else:
        print("The ray is parallel to the plane, no intersection.")
    '''

def transform_plane(plane_coeffs, transformation_matrix):
    # Unpack the plane coefficients
    a, b, c, d = plane_coeffs
    
    # Extract the rotation matrix R and translation vector t from the transformation matrix
    R = transformation_matrix[:3, :3]
    t = transformation_matrix[:3, 3]
    
    # Compute the normal vector in the new referential
    normal_vector = np.array([a, b, c])
    new_normal_vector = np.dot(np.linalg.inv(R).T, normal_vector)
    
    # Calculate the new d' value
    new_d = d - np.dot(new_normal_vector, t)
    
    # Return the new plane coefficients (a', b', c', d')
    return (*new_normal_vector, new_d)