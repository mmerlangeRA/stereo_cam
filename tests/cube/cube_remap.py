from typing import Tuple
import cv2
import numpy as np

from scipy.optimize import least_squares

from src.features_2d.utils import detectAndComputeKPandDescriptors, getMatches
from src.utils.cube_image import get_cube_faces

def map_directions_to_cube_faces(directions):
    """
    Map 3D directions to cube face indices and UV coordinates.

    Parameters:
        directions (np.ndarray): Array of 3D direction vectors (N x 3).

    Returns:
        tuple: face_indices (N,), uv_coords (N x 2)
    """
    abs_dirs = np.abs(directions)
    max_axis = np.argmax(abs_dirs, axis=1)
    signs = np.sign(directions)
    face_indices = np.zeros(len(directions), dtype=np.int32)
    uv_coords = np.zeros((len(directions), 2), dtype=np.float32)

    # Right face (positive X)
    idx = (max_axis == 0) & (signs[:, 0] > 0)
    face_indices[idx] = 0  # Right
    uv_coords[idx] = map_direction_to_uv(directions[idx], axis='x', positive=True)

    # Left face (negative X)
    idx = (max_axis == 0) & (signs[:, 0] < 0)
    face_indices[idx] = 1  # Left
    uv_coords[idx] = map_direction_to_uv(directions[idx], axis='x', positive=False)

    # Top face (positive Y)
    idx = (max_axis == 1) & (signs[:, 1] > 0)
    face_indices[idx] = 2  # Top
    uv_coords[idx] = map_direction_to_uv(directions[idx], axis='y', positive=True)

    # Bottom face (negative Y)
    idx = (max_axis == 1) & (signs[:, 1] < 0)
    face_indices[idx] = 3  # Bottom
    uv_coords[idx] = map_direction_to_uv(directions[idx], axis='y', positive=False)

    # Front face (positive Z)
    idx = (max_axis == 2) & (signs[:, 2] > 0)
    face_indices[idx] = 4  # Front
    uv_coords[idx] = map_direction_to_uv(directions[idx], axis='z', positive=True)

    # Back face (negative Z)
    idx = (max_axis == 2) & (signs[:, 2] < 0)
    face_indices[idx] = 5  # Back
    uv_coords[idx] = map_direction_to_uv(directions[idx], axis='z', positive=False)

    return face_indices, uv_coords

def map_direction_to_uv(directions, axis='x', positive=True):
    """
    Map 3D directions to UV coordinates on a cube face.

    Parameters:
        directions (np.ndarray): Array of 3D direction vectors.
        axis (str): Axis of the cube face ('x', 'y', or 'z').
        positive (bool): Whether the face is in the positive or negative direction.

    Returns:
        np.ndarray: UV coordinates (N x 2)
    """
    if axis == 'x':
        major = directions[:, 0]
        u = (directions[:, 2] / np.abs(major) + 1) / 2
        v = (-directions[:, 1] / np.abs(major) + 1) / 2
        if positive:
            u = 1 - u
    elif axis == 'y':
        major = directions[:, 1]
        u = (-directions[:, 0] / np.abs(major) + 1) / 2
        v = (directions[:, 2] / np.abs(major) + 1) / 2
        if positive:
            v = 1 - v  # Flip V coordinate for positive Y face
    elif axis == 'z':
        major = directions[:, 2]
        u = (-directions[:, 0] / np.abs(major) + 1) / 2
        v = (-directions[:, 1] / np.abs(major) + 1) / 2
        if positive:
            u = 1 - u

    else:
        raise ValueError("Invalid axis")
    return np.stack((u, v), axis=-1)

def cube_to_equirectangular(undistorted_faces, output_size=(2048, 1024)):
    """
    Convert undistorted cube faces to an equirectangular image using cv2.remap.

    Parameters:
        undistorted_faces (dict): Dictionary of undistorted cube face images.
        output_size (tuple): Size of the output equirectangular image (width, height).

    Returns:
        np.ndarray: Equirectangular image.
    """
    width, height = output_size
    map_x = np.zeros((height, width), dtype=np.float32)
    map_y = np.zeros((height, width), dtype=np.float32)

    # Compute longitude and latitude for each pixel
    lon = (np.linspace(0, width - 1, width) / width) * 2 * np.pi - np.pi
    lat = -((np.linspace(0, height - 1, height) / height) * np.pi - np.pi / 2)
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    # Convert spherical coordinates to 3D directions
    x = np.cos(lat_grid) * np.sin(lon_grid)
    y = np.sin(lat_grid)
    z = np.cos(lat_grid) * np.cos(lon_grid)
    directions = np.stack((x, y, z), axis=-1)

    # Map directions to cube faces and UV coordinates
    face_indices, uv_coords = map_directions_to_cube_faces(directions.reshape(-1, 3))
    face_indices = face_indices.reshape(height, width)
    uv_coords = uv_coords.reshape(height, width, 2)

    # Create empty equirectangular image
    equirect_img = np.zeros((height, width, 3), dtype=np.uint8)

    # For each face, remap pixels
    for i, face in enumerate(['right', 'left', 'top', 'bottom', 'front', 'back']):
        idx = face_indices == i
        if np.any(idx):
            face_img = undistorted_faces[face]
            h_face, w_face = face_img.shape[:2]
            # Create maps for remapping
            map_face_x = (uv_coords[:, :, 0] * w_face).astype(np.float32)
            map_face_y = (uv_coords[:, :, 1] * h_face).astype(np.float32)
            # Mask indices
            map_face_x[~idx] = -1
            map_face_y[~idx] = -1
            # Remap pixels
            remapped_face = cv2.remap(face_img, map_face_x, map_face_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            # Assign to equirectangular image
            equirect_img[idx] = remapped_face[idx]

    return equirect_img

if __name__=="__main__":
    imgLeft_name =r"C:\Users\mmerl\projects\stereo_cam\Photos\P1\D_P1_CAM_D_0_CUBE.png"
    imgLeft=cv2.imread(imgLeft_name)
    cube_faces= get_cube_faces(imgLeft)
    for face in cube_faces.keys():
        cv2.imshow(face, cube_faces[face])
        print(cube_faces[face].shape)
    cv2.waitKey(0)
    equirect_img = cube_to_equirectangular(cube_faces, output_size=(5376, 2688))
    cv2.imwrite(r"C:\Users\mmerl\projects\stereo_cam\Photos\P1\equirectangular.png", equirect_img)
    cv2.imshow('equirectangular', equirect_img)

    cv2.waitKey(0)


