import os
import numpy as np
import cv2 as cv
from scipy.optimize import minimize
from src.calibrate import calibrate_left_right
from src.triangulate import rotation_matrix_from_params, triangulate_point,get_3d_point_cam1_2_from_coordinates


def estimate_pos_on_screen(d1,d2, t,image_width,image_height):
    cosA =(pow(d1,2)+pow(t,2)-pow(d2,2))/(2*d1*t)
    cosB =(pow(d2, 2)+pow(t, 2)-pow(d1, 2))/(2*d2*t)
    A  = np.arccos(cosA)
    B = np.arccos(cosB)
    print(A, B)
    theta1 = np.pi/2.-A
    theta2 = B - np.pi/2.
    print(theta1, theta2)

    normalizedTheta1 = theta1/np.pi
    normalizedTheta2 = theta2/np.pi

    half_width = image_width/2.

    posx1= normalizedTheta1*half_width+half_width
    posx2= normalizedTheta2*half_width+half_width
    posx1_b= 0
    posx2_b= 0
    return posx1, posx2,posx1_b,posx2_b

tx=1.12
print("****************************************")
posx1, posx2,posx1_b,posx2_b=estimate_pos_on_screen(14,14,tx, 5376,200)
print(posx1, posx2,posx1_b,posx2_b)
# print("****************************************")
# posx1, posx2,posx1_b,posx2_b=estimate_pos_on_screen(14,14,tx, 5376,200)
# print(posx1, posx2,posx1_b,posx2_b)


def objective_function(params, ray1, ray2, known_distance):
    """Objective function for optimization."""
    R_matrix = rotation_matrix_from_params(params[:3])
    tx = params[3:][0]
    point_3d,residuals = triangulate_point(ray1, ray2, tx, R_matrix)
    estimated_distance = np.linalg.norm(point_3d)
    return (estimated_distance - known_distance) ** 2

def global_objective_function(params, data):
    target =0
    R_matrix = rotation_matrix_from_params(params[:3])
    tx = params[3:][0]
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
        point_3d_1,residuals = triangulate_point(ray1, ray2, tx,R_matrix)
        distance_cam1_to_point = np.linalg.norm(point_3d_1)
        point_3d_cam2 = R_matrix @ (point_3d_1 - [0.,tx,0.])  
        distance_cam2_to_point = np.linalg.norm(point_3d_cam2)
        target += (distance_cam1_to_point - known_distance_1) ** 2
        target += (distance_cam2_to_point - known_distance_2) ** 2
    return target


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

            "keypoints": [[3206, 1223],[3046, 1333]],
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
        {#P3_1

            "keypoints": [[3043, 1316],[2887, 1389]],
            "distances": [11.08,10.6],
            "image_width": 5376,
            "image_height": 2688,
            "valid":True
        },
         {#P4_1

            "keypoints": [[2825, 1405],[2705, 1422]],
            "distances": [19.6,19.6],
            "image_width": 5376,
            "image_height": 2688,
            "valid":True
        },
        {#P4_2

            "keypoints": [[2878, 1458],[2869, 1473]],
            "distances": [8.379,8.179],
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
# Main execution
img_folder = os.path.join(os.getcwd(), 'Photos', 'P1')
left_image_path = 'D_P1_CAM_G_2_EAC.png'
right_image_path = 'D_P1_CAM_D_2_EAC.png'

left_image = cv.imread(os.path.join(img_folder, left_image_path))
right_image = cv.imread(os.path.join(img_folder, right_image_path))

# Image dimensions
image_height,image_width = left_image.shape[:2]

#top right of T
point_image1 = np.array([3206, 1223])  # Keypoints in image from Camera 1 (left)
point_image2 = np.array([3046, 1333])  # Keypoints in image from Camera 2 (right)

# Relative position and rotation (example values)
t = [1.12,0,0]  # Example translation: 1.12 meters along the x-axis
R = np.eye(3)  # Assuming no rotation for simplicity

# Triangulate the 3D point
p1,p2,_=get_3d_point_cam1_2_from_coordinates(point_image1, point_image2, image_width, image_height, R, t)
print("p1", p1,np.linalg.norm(p1))
print("p2", p2,np.linalg.norm(p2))

best_results  = calibrate_left_right(left_image,right_image)
print(best_results)

'''
p1,p2=get_3d_point_cam1_2_from_coordinates(point_image1, point_image2, image_width, image_height, R, tx+0.01)
print("p1", p1,np.linalg.norm(p1))
print("p2", p2,np.linalg.norm(p2))
#1cm => 5 cm
'''

if False:
    average = 0
    for d in data:
        print("#####Checking########")
        point_image1 = np.array(d["keypoints"][0])
        point_image2 = np.array(d["keypoints"][1])
        d1=d["distances"][0]
        d2 =d["distances"][1]
        image_width = d["image_width"]
        image_height = d["image_height"]
        posx1,posx2,posx1_b,posx2_b = estimate_pos_on_screen(d1,d2, tx,image_width,image_height)

        minx1 = min(abs(posx1-point_image1[0]),abs(posx1_b-point_image1[0]))
        minx2 = min(abs(posx2-point_image2[0]),abs(posx2_b-point_image2[0]))
        m = max(minx1, minx2)
        average +=m
        if m > 50:
            print("expected",point_image1[0],point_image2[0])
            print(f"from {d1},{d2},{tx}")
            print(posx1,posx2,posx1_b,posx2_b )
            print("error?", m)
        print(point_image1,point_image2)
    print("average", average/len(data))

if False:
    angle_max = np.pi*7./180.
    bnds = ((-angle_max, angle_max), (-angle_max, angle_max),(-angle_max, angle_max),(1.05, 1.19))

    # Optimize for the rotation parameters
    print("######### Optimize 1############")
    initial_params = [0, 0, 0, 1.12]
    known_distance = 9.68
    result = minimize(objective_function, initial_params, args=(ray1, ray2, known_distance), method='BFGS')

    # Extract the optimized rotation matrix
    optimized_params = result.x
    optimized_R = rotation_matrix_from_params(optimized_params[:3])
    optimized_tx = optimized_params[3:][0]
    print("optimized_tx",optimized_tx)

    point_3d,residuals = triangulate_point(ray1, ray2, optimized_tx, optimized_R)

    print("Optimized rotation matrix R:\n", optimized_R)
    euler_angles_rad = SciPyRotation.from_matrix(optimized_R).as_euler('xyz')
    euler_angles_deg = np.degrees(euler_angles_rad)
    print("Rotation angles (degrees):\n", euler_angles_deg)

    print("Optimized t:\n", optimized_tx)
    print("3D coordinates of the point:", point_3d)
    print(f'optimized with distance {np.linalg.norm(point_3d)}')

    print("######### Optimize Global############")
    initial_params = [0, 0, 0, 1.12]
    result = minimize(global_objective_function, initial_params, args=(data), bounds=bnds)
    # Extract the optimized rotation matrix
    optimized_params = result.x
    optimized_R = rotation_matrix_from_params(optimized_params[:3])
    optimized_tx = optimized_params[3:][0]
    print("optimized_tx",optimized_tx)

    point_3d,residuals = triangulate_point(ray1, ray2, optimized_tx, optimized_R)

    print("Optimized rotation matrix R:\n", optimized_R)
    euler_angles_rad = SciPyRotation.from_matrix(optimized_R).as_euler('xyz')
    euler_angles_deg = np.degrees(euler_angles_rad)
    print("Rotation angles (degrees):\n", euler_angles_deg)
    print("Optimized t:\n", optimized_tx)
    print("3D coordinates of the point:", point_3d)
    print(f'optimized with distance {np.linalg.norm(point_3d)}')

    # Calculate the position of the 3D point relative to the second camera
    point_3d_cam2 = optimized_R @ (point_3d - optimized_tx)  # Rotate and translate the point to the second camera's coordinate system
    distance_cam2_to_point = np.linalg.norm(point_3d_cam2)

    # Output the distance
    print("Distance from the second camera to the 3D point:", distance_cam2_to_point)

if False:
    optimized_R= best_results["R"]
    optimized_t= best_results["t"]
    for d in data:
        if not d["valid"]:
            continue
        p1,p2,_ = get_3d_point_cam1_2_from_coordinates(d["keypoints"][0], d["keypoints"][1], image_width, image_height, optimized_R, optimized_t)
        print(p1,p2)
        print(f'{d["distances"][0]} vs {np.linalg.norm(p1)}, {d["distances"][1]} vs {np.linalg.norm(p2)}')

