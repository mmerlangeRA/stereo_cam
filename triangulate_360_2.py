import os
import numpy as np
import cv2 as cv
from scipy.optimize import minimize
from src.calibrate import calibrate_left_right, getCalibrationFrom3Matching, load_calibration_custom, load_calibration_params, save_calibration_custom, save_calibration_params
from src.triangulate import rotation_matrix_from_params, triangulate_point,get_3d_point_cam1_2_from_coordinates
import yaml


ray1 = np.array([1.1, 0.1, 0.2])
ray2 = np.array([0, 1.2, 0.1])
identity_t = np.array([0, 0, 0])
identity_R = np.eye(3)


''' "good one"
{'max_inliers': 2232, 'R': array([[ 0.99660993,  0.01121534,  0.08150376],
       [-0.01297537,  0.99969323,  0.02109695],
       [-0.08124215, -0.02208297,  0.99644973]]), 't': array([ 1.08000034, -0.01266376, -0.05783629])}
[4.8285099  0.44539063 7.78940202] [4.38051853 0.57482834 7.50472628] [6.35491574e-06]
[ 4.80901366 -0.30651955  7.74210788] [ 4.3488008  -0.17759597  7.47578838] [9.73542891e-06]
size 0.7536482982240809
'''

'''
R [[ 0.99986244  0.00424974 -0.01603238]
 [-0.00395075  0.99981855  0.01863503]
 [ 0.01610867 -0.01856913  0.9996978 ]]
t [ 1.08        0.01091257 -0.1       ]
'''

data=[
        {#P1_0
            "id":"P1_0",
            "keypoints_top": [[522, 1278],[512, 1267]],
            "keypoints_bottom": [[522, 1327],[509, 1315]],
            "distances": [13.8,14.5],
            "image_width": 5376,
            "image_height": 2688,
            "valid":True
        },
        {#P1_1
            "id":"P1_1",
            "keypoints_top": [[803, 1258],[790, 1258]],
            "keypoints_bottom": [[801, 1330],[786, 1323]],
            "distances": [9.57,10.6],
            "image_width": 5376,
            "image_height": 2688,
            "valid":True
        },
        {#P1_2
            "id":"P1_2",
            "keypoints_top": [[3196, 1297],[3034, 1305]],
            "keypoints_bottom": [[3196, 1367],[3035, 1379]],
            "distances": [9.68,9.05],
            "image_width": 5376,
            "image_height": 2688,
            "valid":True
        },
        {#P2_2
            "id":"P2_2",
            "keypoints_top": [[2824, 1405],[2672, 1492]],
            "keypoints_bottom": [[2824, 1436],[2670, 1561]],
            "distances": [6.35,6.28],#probably mistake here, so I switched, obviously panneau à gauche
            "image_width": 5376,
            "image_height": 2688,
            "valid":True
        },
        {#P2_1
            "id":"P2_1",
            "keypoints": [[2820, 1397],[2703, 1436]],
            "distances": [13.85,13.77],#probably mistake here, so I switched, obviously panneau à droite
            "image_width": 5376,
            "image_height": 2688,
            "valid":False
        },
        {#P3_1
            "id":"P3_1",
            "keypoints_top": [[3007, 1273],[2848, 1285]],
            "keypoints_bottom": [[3075, 1272],[2922, 1284]],
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
            "valid":False
        },
        {#P4_2

            "keypoints": [[2878, 1458],[2869, 1473]],
            "distances": [8.379,8.179],
            "image_width": 5376,
            "image_height": 2688,
            "valid":False
        },
        {#P5_2
            "id":"P5_2",
            "keypoints_top": [[3034, 1258],[2851, 1268]],
            "keypoints_bottom": [[3035, 1329],[2853, 1342]],
            "distances": [8.8,8.35],
            "image_width": 5376,
            "image_height": 2688,
            "valid":True
        }
    ]
# Main execution
img_folder = os.path.join(os.getcwd(), 'Photos', 'P1')
left_image_path = 'D_P1_CAM_G_1_EAC.png'
right_image_path = 'D_P1_CAM_D_1_EAC.png'

left_image = cv.imread(os.path.join(img_folder, left_image_path))
right_image = cv.imread(os.path.join(img_folder, right_image_path))

# Image dimensions
image_height,image_width = left_image.shape[:2]

angle_max = np.pi*10./180.
dt_max = 0.12001
bnds = ((-angle_max, angle_max), (-angle_max, angle_max),(-angle_max, angle_max),(1.11, 1.13),(-dt_max,dt_max),(-dt_max,dt_max))
initial_params = [0, 0, 0, 1.12,0,0]
initial_params_0 = [0, 0, 0, 0.0,0,0]

identity_t = np.array([0, 0, 0])
identity_R = np.eye(3)

initial_t=np.array([0.12, 0, 0])

''' for P1_2
optimized_R [[ 9.99999992e-01 -1.25840390e-04  1.97564767e-05]
 [ 1.25841868e-04  9.99999989e-01 -7.48031043e-05]
 [-1.97470632e-05  7.48055899e-05  9.99999997e-01]]
R [[ 9.99990405e-01  4.37535532e-03 -2.15538834e-04]
 [-4.37556379e-03  9.99989950e-01 -9.76420750e-04]
 [ 2.11264480e-04  9.77354486e-04  9.99999500e-01]]
t [ 1.11911477  0.09601804 -0.12001   ]
'''

'''for P1_1
R [[ 9.99989311e-01  4.61184898e-03 -3.29632740e-04]
 [-4.60813155e-03  9.99934190e-01  1.05061928e-02]
 [ 3.78064021e-04 -1.05045615e-02  9.99944754e-01]]
t [1.11    0.12001 0.12001]

'''

#point_3d, residuals = triangulate_point(ray1, ray2, t, R, verbose=True)
#print("3D coordinates of the point:", point_3d)
#print("Residuals:", residuals)
sub_uv1 = [[3196, 1297]]
sub_uv2 = [[3034, 1305]]

print("********If identity******")
p1_top,p2_top,residual_distance_normalized = get_3d_point_cam1_2_from_coordinates(sub_uv1[0], sub_uv2[0], image_width, image_height, identity_R, initial_t,True)
print(p1_top,p2_top,residual_distance_normalized)
print("\n")

print("********If hard coded ******")
optimized_R_s = np.array([[ 0.99660993,  0.01121534,  0.08150376],
    [-0.01297537,  0.99969323,  0.02109695],
    [-0.08124215, -0.02208297,  0.99644973]])
optimized_t_s=np.array([ 1.08000034, -0.01266376, -0.05783629])
print("R", optimized_R_s)
print("t", optimized_t_s)
p1_top,p2_top,residual_distance_normalized = get_3d_point_cam1_2_from_coordinates(sub_uv1[0], sub_uv2[0], image_width, image_height, optimized_R_s, optimized_t_s,True)
print(p1_top,p2_top,residual_distance_normalized)
print("\n")

print("********Optimizing 1 point only******")
optimized_params, residual_distance_normalized = getCalibrationFrom3Matching(sub_uv1, sub_uv2, initial_params,left_image.shape[1], left_image.shape[0],bnds)
print("optimized_params", optimized_params)
print("residual_distance_normalized", residual_distance_normalized)
optimized_R = rotation_matrix_from_params(optimized_params[:3])
optimized_t = optimized_params[3:]

p1_top,p2_top,residual_distance_normalized = get_3d_point_cam1_2_from_coordinates(sub_uv1[0], sub_uv2[0], image_width, image_height, optimized_R, optimized_t,True)
print(p1_top,p2_top,residual_distance_normalized)

print("p1_top",p1_top)
print("p2_top",p2_top)
print(f'p1_top size {np.linalg.norm(p1_top)}')
print(f'distance {np.linalg.norm(p1_top-p2_top)}')

print("\n")

if True:
    regenerate = True  
    file_name= "3_1"
    s = file_name.split("_")
    photo = s[0]
    angle = s[1]
    img_folder = os.path.join(os.getcwd(), 'Photos', 'P'+photo)
    left_image_path = 'D_P'+photo+'_CAM_G_'+angle+'_EAC.png'
    right_image_path = 'D_P'+photo+'_CAM_D_'+angle+'_EAC.png'
    print(left_image_path, right_image_path)

    left_image = cv.imread(os.path.join(img_folder, left_image_path))
    right_image = cv.imread(os.path.join(img_folder, right_image_path))
    print("********Full calibration******")
    if regenerate or not os.path.exists(file_name+".npz"):
        best_results = calibrate_left_right(left_image, right_image, initial_params, bnds)
        optimized_R= best_results["R"]
        optimized_t= best_results["t"]
        save_calibration_params(best_results["params"],file_name)
    else:
        optimized_params = load_calibration_params(file_name)
        optimized_R = rotation_matrix_from_params(optimized_params[:3])
        optimized_t = optimized_params[3:]

    print("R", optimized_R)
    print("t", optimized_t)
    print("\n")
                                                             

if True:
    for d in data:
        if not d["valid"]:
            continue
        print("----- Calibrated R, T -------------------")
        print(d["id"])
        p1_top,p2_top,residual_distance_normalized = get_3d_point_cam1_2_from_coordinates(d["keypoints_top"][0], d["keypoints_top"][1], image_width, image_height, optimized_R, optimized_t, False)
        print(p1_top,p2_top,residual_distance_normalized)
        p1_bottom,p2_bottom,residual_distance_normalized = get_3d_point_cam1_2_from_coordinates(d["keypoints_bottom"][0], d["keypoints_bottom"][1], image_width, image_height, optimized_R, optimized_t)
        print(p1_bottom,p2_bottom,residual_distance_normalized)
        print(f'panneau size {np.linalg.norm(p1_top-p1_bottom)}')
        print(f'{d["distances"][0]} vs {np.linalg.norm(p1_top)}, {d["distances"][1]} vs {np.linalg.norm(p2_top)}')
        print("\n")
        if False:
            print("----- Hard coded R, T -------------------")
            p1_top,p2_top,residual_distance_normalized = get_3d_point_cam1_2_from_coordinates(d["keypoints_top"][0], d["keypoints_top"][1], image_width, image_height, optimized_R_s, optimized_t_s,False)
            print(p1_top,p2_top,residual_distance_normalized)
            p1_bottom,p2_bottom,residual_distance_normalized = get_3d_point_cam1_2_from_coordinates(d["keypoints_bottom"][0], d["keypoints_bottom"][1], image_width, image_height, optimized_R_s, optimized_t_s)
            print(p1_bottom,p2_bottom,residual_distance_normalized)
            print(f'size {np.linalg.norm(p1_top-p1_bottom)}')
            print(f'{d["distances"][0]} vs {np.linalg.norm(p1_top)}, {d["distances"][1]} vs {np.linalg.norm(p2_top)}')

