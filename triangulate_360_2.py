import os
import numpy as np
import cv2 as cv
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as SciPyRotation


def estimate_pos_on_screen(d1,d2, t,image_width,image_height):
    return 0

def rotation_matrix_from_params(params):
    """Construct a rotation matrix from parameters."""
    return SciPyRotation.from_euler('xyz', params,degrees=False).as_matrix()

def objective_function(params, ray1, ray2, known_distance):
    """Objective function for optimization."""
    R_matrix = rotation_matrix_from_params(params[:3])
    ty = params[3:][0]
    point_3d = triangulate_point(ray1, ray2, ty, R_matrix)
    estimated_distance = np.linalg.norm(point_3d)
    return (estimated_distance - known_distance) ** 2

def global_objective_function(params, data):
    target =0
    R_matrix = rotation_matrix_from_params(params[:3])
    ty = params[3:][0]
    #for d in data:
    for d in data:
        if not d["valid"]:
            continue
        image_width = d["image_width"]
        image_height = d["image_height"]
        d["keypoints"][0], d["keypoints"][1]
        point_image1 = np.array( d["keypoints"][0]) 
        point_image2 = np.array(d["keypoints"][1]) 
        known_distance_1 = d["distances"][0]
        known_distance_2 = d["distances"][1]
        theta1, phi1 = pixel_to_spherical_revised(image_width, image_height, point_image1[0], point_image1[1])
        theta2, phi2 = pixel_to_spherical_revised(image_width, image_height, point_image2[0], point_image2[1])
        ray1 = spherical_to_cartesian(theta1, phi1)
        ray2 = spherical_to_cartesian(theta2, phi2)
        point_3d_1 = triangulate_point(ray1, ray2, ty,R_matrix)
        distance_cam1_to_point = np.linalg.norm(point_3d_1)
        point_3d_cam2 = R_matrix @ (point_3d_1 - [0.,ty,0.])  
        distance_cam2_to_point = np.linalg.norm(point_3d_cam2)
        target += (distance_cam1_to_point - known_distance_1) ** 2
        target += (distance_cam2_to_point - known_distance_2) ** 2
    return target

def pixel_to_spherical(image_width, image_height, pixel_x, pixel_y):
    """Convert pixel coordinates to spherical coordinates (theta, phi)."""
    theta = (pixel_x / image_width) * 2 * np.pi
    phi = (pixel_y / image_height) * np.pi
    return theta, phi

def pixel_to_spherical_revised(image_width, image_height, pixel_x, pixel_y):
    """Convert pixel coordinates to spherical coordinates (theta, phi)."""
    theta = (pixel_x / image_width) * 2 * np.pi - np.pi #longitude
    phi = (pixel_y / image_height) * np.pi - np.pi / 2 #latitude
    return theta, phi

def spherical_to_cartesian(theta, phi):
    """Convert spherical coordinates to 3D cartesian coordinates."""
    x = np.cos(phi) * np.sin(theta)
    y = np.sin(phi)
    z = np.cos(phi) * np.cos(theta)
    return np.array([x, y, z])

def triangulate_point(ray1, ray2, ty:float, R_matrix,verbose=False):
    """Triangulate a 3D point from two rays and the relative camera transformation."""
    # Formulate the system of linear equations
    A = np.vstack((ray1, -R_matrix @ ray2,np.cross(ray1, R_matrix @ ray2))).T
    b = np.vstack((0.,ty,0.))
    if verbose:
        print("triangulate_point")
        print("ty",ty)
        print("ray1",ray1)
        print("ray2",ray2)
        print("A",A)
        print("b",b)
    
    # Solve for lambda1 and lambda2
    lambdas, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    lambda1, lambda2,_ = lambdas

    print("test",A @ lambdas)
    
    # Calculate the 3D point using lambda1
    point_3d = lambda1 * ray1
    return point_3d

def get_3d_point_cam1_2_from_coordinates(keypoints_cam1, keypoints_cam2,image_width, image_height,R, t):
    point_image1 = np.array(keypoints_cam1) 
    point_image2 = np.array(keypoints_cam2) 
    theta1, phi1 = pixel_to_spherical_revised(image_width, image_height, point_image1[0], point_image1[1])
    theta2, phi2 = pixel_to_spherical_revised(image_width, image_height, point_image2[0], point_image2[1])
    ray1 = spherical_to_cartesian(theta1, phi1)
    ray2 = spherical_to_cartesian(theta2, phi2)
    point_3d_cam1 = triangulate_point(ray1, ray2, t, R)
    point_3d_cam2 = R @ (point_3d - t) 
    return point_3d_cam1,point_3d_cam2

data=[
        {#P1
            "keypoints": [[537, 1303],[525, 1292]],
            "distances": [13.8,14.5],
            "image_width": 5376,
            "image_height": 2688,
            "valid":False
        },
        {
            "keypoints": [[823, 1296],[806, 1292]],
            "distances": [9.57,10.6],
            "image_width": 5376,
            "image_height": 2688,
            "valid":False
        },
        {#P1_2

            "keypoints": [[3217, 1331],[3055, 1340]],
            "distances": [9.68,9.05],#probably mistake here, so I switched, obviously panneau à droite
            "image_width": 5376,
            "image_height": 2688,
            "valid":True
        },
        {#P2_2

            "keypoints": [[2843, 1411],[2717, 1505]],
            "distances": [6.35,6.28],#probably mistake here, so I switched, obviously panneau à gauche
            "image_width": 5376,
            "image_height": 2688,
            "valid":False
        },
        {#P2_1

            "keypoints": [[2820, 1397],[2703, 1436]],
            "distances": [13.85,13.77],#probably mistake here, so I switched, obviously panneau à droite
            "image_width": 5376,
            "image_height": 2688,
            "valid":True
        },
        {#P5_2

            "keypoints": [[3066, 1283],[2886, 1293]],
            "distances": [8.8,8.35],
            "image_width": 5376,
            "image_height": 2688,
            "valid":True
        }
    ]

angle_max = np.pi*15./180.

bnds = ((-angle_max, angle_max), (-angle_max, angle_max),(-angle_max, angle_max),(1.05, 1.19))
# Main execution
img_folder = os.path.join(os.getcwd(), 'Photos', 'P1')
left_image_path = 'D_P1_CAM_G_2_EAC.png'
right_image_path = 'D_P1_CAM_D_2_EAC.png'

left_image = cv.imread(os.path.join(img_folder, left_image_path))
right_image = cv.imread(os.path.join(img_folder, right_image_path))

# Image dimensions
image_height,image_width = left_image.shape[:2]

'''
point_image1 = np.array([0, 1344])  # Keypoints in image from Camera 1 (left)
theta1, phi1 = pixel_to_spherical_revised(image_width, image_height, point_image1[0], point_image1[1])
ray1 = spherical_to_cartesian(theta1, phi1)
print(point_image1)
print(theta1, phi1)
print(ray1)
point_image1 = np.array([2688, 1344])  # Keypoints in image from Camera 1 (left)
theta1, phi1 = pixel_to_spherical_revised(image_width, image_height, point_image1[0], point_image1[1])
ray1 = spherical_to_cartesian(theta1, phi1)
print(point_image1)
print(theta1, phi1)
print(ray1)
point_image1 = np.array([5376, 1344])  # Keypoints in image from Camera 1 (left)
theta1, phi1 = pixel_to_spherical_revised(image_width, image_height, point_image1[0], point_image1[1])
ray1 = spherical_to_cartesian(theta1, phi1)
print(point_image1)
print(theta1, phi1)
print(ray1)
point_image1 = np.array([2688, 0])  # Keypoints in image from Camera 1 (left)
theta1, phi1 = pixel_to_spherical_revised(image_width, image_height, point_image1[0], point_image1[1])
ray1 = spherical_to_cartesian(theta1, phi1)
print(point_image1)
print(theta1, phi1)
print(ray1)
'''

#center letter O
point_image1 = np.array([3217, 1331])  # Keypoints in image from Camera 1 (left)
point_image2 = np.array([3055, 1340])  # Keypoints in image from Camera 2 (right)

# Convert pixel coordinates to spherical coordinates
theta1, phi1 = pixel_to_spherical_revised(image_width, image_height, point_image1[0], point_image1[1])
theta2, phi2 = pixel_to_spherical_revised(image_width, image_height, point_image2[0], point_image2[1])
#theta is longitude
#print(theta1,phi1)
# Convert spherical coordinates to 3D cartesian coordinates
ray1 = spherical_to_cartesian(theta1, phi1)
ray2 = spherical_to_cartesian(theta2, phi2)

# Relative position and rotation (example values)
ty = 1.12  # Example translation: 1.12 meters along the x-axis
R = np.eye(3)  # Assuming no rotation for simplicity

# Triangulate the 3D point
point_3d = triangulate_point(ray1, ray2, ty, R,True)
print("3D coordinates of the point:", point_3d)
print(np.linalg.norm(point_3d))

if False:
    # Optimize for the rotation parameters
    print("######### Optimize 1############")
    initial_params = [0, 0, 0, 1.12]
    known_distance = 9.05
    result = minimize(objective_function, initial_params, args=(ray1, ray2, known_distance), method='BFGS')

    # Extract the optimized rotation matrix
    optimized_params = result.x
    optimized_R = rotation_matrix_from_params(optimized_params[:3])
    optimized_ty = optimized_params[3:][0]
    print("optimized_ty",optimized_ty)

    point_3d = triangulate_point(ray1, ray2, optimized_ty, optimized_R)

    print("Optimized rotation matrix R:\n", optimized_R)
    euler_angles_rad = SciPyRotation.from_matrix(optimized_R).as_euler('xyz')
    euler_angles_deg = np.degrees(euler_angles_rad)
    print("Rotation angles (degrees):\n", euler_angles_deg)

    print("Optimized t:\n", optimized_ty)
    print("3D coordinates of the point:", point_3d)
    print(f'optimized with distance {np.linalg.norm(point_3d)}')

    print("######### Optimize Global############")
    initial_params = [0, 0, 0, 1.12]
    result = minimize(global_objective_function, initial_params, args=(data), method='SLSQP',bounds=bnds)
    # Extract the optimized rotation matrix
    optimized_params = result.x
    optimized_R = rotation_matrix_from_params(optimized_params[:3])
    optimized_ty = optimized_params[3:][0]
    print("optimized_ty",optimized_ty)

    point_3d = triangulate_point(ray1, ray2, optimized_ty, optimized_R)

    print("Optimized rotation matrix R:\n", optimized_R)
    euler_angles_rad = SciPyRotation.from_matrix(optimized_R).as_euler('xyz')
    euler_angles_deg = np.degrees(euler_angles_rad)
    print("Rotation angles (degrees):\n", euler_angles_deg)
    print("Optimized t:\n", optimized_ty)
    print("3D coordinates of the point:", point_3d)
    print(f'optimized with distance {np.linalg.norm(point_3d)}')

    # Calculate the position of the 3D point relative to the second camera
    point_3d_cam2 = optimized_R @ (point_3d - optimized_ty)  # Rotate and translate the point to the second camera's coordinate system
    distance_cam2_to_point = np.linalg.norm(point_3d_cam2)

    # Output the distance
    print("Distance from the second camera to the 3D point:", distance_cam2_to_point)

    for d in data:
        if not d["valid"]:
            continue
        p1,p2 = get_3d_point_cam1_2_from_coordinates(d["keypoints"][0], d["keypoints"][1], image_width, image_height, optimized_R, optimized_ty)
        print(p1,p2)
        print(f'{d["distances"][0]} vs {np.linalg.norm(p1)}, {d["distances"][1]} vs {np.linalg.norm(p2)}')

