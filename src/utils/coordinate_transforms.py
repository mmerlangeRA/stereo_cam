from typing import Tuple
import numpy as np
import numpy.typing as npt
import cv2


def pixel_to_spherical(image_width: int, image_height: int, pixel_x: int, pixel_y: int) -> Tuple[float, float]:
    """Convert pixel coordinates to spherical coordinates (theta, phi)."""
    theta = (pixel_x / image_width) * 2 * np.pi - np.pi # longitude
    phi = (pixel_y / image_height) * np.pi - np.pi / 2 # latitude
    return theta, phi

def spherical_to_cartesian(theta: float, phi: float) -> np.ndarray:
    """Convert spherical coordinates to 3D cartesian coordinates."""
    x = np.cos(phi) * np.sin(theta)
    y = np.sin(phi)
    z = np.cos(phi) * np.cos(theta)
    return np.array([x, y, z])

def cartesian_to_spherical(x:float,y:float,z:float) ->  Tuple[float, float,float]:
    """Convert 3D cartesian coordinates to spherical coordinates ."""
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(z, x)
    phi = np.arcsin(y / r)
    return np.array([r,theta, phi])

def spherical_to_equirectangular(theta:float, phi:float, image_width:int, image_height:int) -> Tuple[int, int]:
    """Convert spherical coordinates to equirectangular pixel coordinates."""
    u = (theta + np.pi) / (2 * np.pi) * image_width
    v = (phi + np.pi / 2) / np.pi * image_height
    return int(u), int(v)

def cartesian_to_equirectangular(x:float, y:float, z:float, image_width:int, image_height:int) -> Tuple[int, int]:
    """Convert 3D cartesian coordinates to equirectangular pixel coordinates."""
    _,theta, phi = cartesian_to_spherical(x, y, z)
    return spherical_to_equirectangular(theta, phi, image_width, image_height)

def cartesian_to_spherical_array(
        x:npt.NDArray[np.float_], 
        y:npt.NDArray[np.float_], 
        z:npt.NDArray[np.float_]
        )-> Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_], npt.NDArray[np.float_]]:
    """Convert 3D cartesian coordinates to spherical coordinates for arrays."""
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(x, z) # Azimuth angle
    phi = np.arcsin(y / r)    # Elevation angle
    return r, theta, phi

def spherical_to_equirectangular_array(
    theta: npt.NDArray[np.float_],
    phi: npt.NDArray[np.float_],
    image_width: int,
    image_height: int
) -> Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
    """
    Convert spherical coordinates to equirectangular pixel coordinates for arrays.

    Parameters:
    - theta: numpy array of azimuth angles in radians.
    - phi: numpy array of elevation angles in radians.
    - image_width: The width of the equirectangular image in pixels.
    - image_height: The height of the equirectangular image in pixels.

    Returns:
    - u: numpy array of horizontal pixel coordinates.
    - v: numpy array of vertical pixel coordinates.
    """
    u = ((theta + np.pi)/(2.*np.pi)) * image_width
    v = (np.pi / 2 - phi) / np.pi * image_height  # Adjusted for image coordinate system
    return u, v

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