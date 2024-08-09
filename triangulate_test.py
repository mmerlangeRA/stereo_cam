import csv
import os
import numpy as np
import cv2
from src.utils.path_utils import save_calibration_params,load_calibration_params
from src.calibration.eac import calibrate_left_right
from src.triangulate.main import get_3d_point_cam1_2_from_coordinates, rotation_matrix_from_params

#Data for estimating
#keypoints_top : coordinates in camLeft and camRight of a "top" point on the sign
#keypoints_bottom : coordinates in camLeft and camRight of a "bottom" point on the sign
data=[
        {#P1_0
            "id":"1_0",
            "keypoints_top": [[522, 1278],[512, 1267]],
            "keypoints_bottom": [[522, 1327],[509, 1315]],
            "distances": [13.8,14.5],
            "height":2.25,
            "image_width": 5376,
            "image_height": 2688,
            "valid":True
        },
        {#P1_1
            "id":"1_1",
            "keypoints_top": [[803, 1258],[790, 1258]],
            "keypoints_bottom": [[801, 1330],[786, 1323]],
            "distances": [9.57,10.6],
            "height":2.25,
            "image_width": 5376,
            "image_height": 2688,
            "valid":True
        },
        {#P1_2
            "id":"1_2",
            "keypoints_top": [[3196, 1297],[3034, 1305]],
            "keypoints_bottom": [[3196, 1367],[3035, 1379]],
            "distances": [9.68,9.05],
            "height":2.25,
            "image_width": 5376,
            "image_height": 2688,
            "valid":True
        },
        {
            "id":"2_1",
            "keypoints_top": [[2824, 1406],[2682, 1429]],
            "keypoints_bottom": [[2824, 1436],[2682, 1462]],
            "distances": [13.77,13.85],
            "height":1.11,
            "image_width": 5376,
            "image_height": 2688,
            "valid":True
        },
        {#P2_2
            "id":"2_2",
            "keypoints_top": [[2896, 1469],[2672, 1492]],
            "keypoints_bottom": [[2894, 1536],[2670, 1561]],
            "distances": [6.35,6.28],#probably mistake here, so I switched, obviously panneau à gauche
            "height":1.11,
            "image_width": 5376,
            "image_height": 2688,
            "valid":True
        },
        {#P3_1
            "id":"3_1",
            "keypoints_top": [[3042, 1272],[2885, 1283]],
            "keypoints_bottom": [[3044, 1335],[2888, 1349]],
            "distances": [11.08,10.6],
            "height":3.05,
            "image_width": 5376,
            "image_height": 2688,
            "valid":True
        },
         {#P4_1
            "id":"4_1",
            "keypoints_top": [[2822, 1395],[2703, 1412]],
            "keypoints_bottom": [[2822, 1424],[2703, 1442]],
            "distances": [19.6,19.6],
            "height":1.22,
            "image_width": 5376,
            "image_height": 2688,
            "valid":True
        },
        {#P4_2
            "id":"4_2",
            "keypoints_top": [[2867, 1434],[2679, 1451]],
            "keypoints_bottom": [[2870, 1504],[2682, 1522]],
            "distances": [8.379,8.179],
            "height":1.22,
            "image_width": 5376,
            "image_height": 2688,
            "valid":True
        },
        {
            "id":"5_1",
            "keypoints_top": [[2884, 1321],[2763, 1336]],
            "keypoints_bottom": [[2884, 1351],[2763, 1367]],
            "distances": [19.26,19.41],
            "height":2.98,
            "image_width": 5376,
            "image_height": 2688,
            "valid":True
        },
        {#P5_2
            "id":"5_2",
            "keypoints_top": [[3034, 1258],[2851, 1268]],
            "keypoints_bottom": [[3035, 1329],[2853, 1342]],
            "distances": [8.8,8.35],
            "height":2.98,
            "image_width": 5376,
            "image_height": 2688,
            "valid":True
        }
    ]

#initial params and bounds
angle_max = np.pi*10./180.
dt_max = 0.12001
bnds = ((-angle_max, angle_max), (-angle_max, angle_max),(-angle_max, angle_max),(1.11, 1.13),(-dt_max,dt_max),(-dt_max,dt_max))
initial_params = [0, 0, 0, 1.12,0,0]

def generate_all_calibrations(initial_params,bnds,inlier_threshold,id):
    print("generate_all_calibrations")
    array_calibration = {}
    for photo in range(0,6):
        for angle in range(0, 3):
            img_folder = os.path.join(os.getcwd(), 'Photos', 'P'+str(photo))
            left_image_path = 'D_P'+str(photo)+'_CAM_G_'+str(angle)+'_EAC.png'
            right_image_path = 'D_P'+str(photo)+'_CAM_D_'+str(angle)+'_EAC.png'
            left_image_path = os.path.join(img_folder, left_image_path)
            right_image_path = os.path.join(img_folder, right_image_path)
            if os.path.exists(left_image_path) and os.path.exists(right_image_path):
                print(f"calibrating {photo}_{angle}")
                left_image = cv2.imread(os.path.join(img_folder, left_image_path))
                right_image = cv2.imread(os.path.join(img_folder, right_image_path))
                best_results = calibrate_left_right(left_image, right_image, initial_params, bnds,inlier_threshold)
                optimized_params = best_results["params"]
                array_calibration[str(photo)+"_"+str(angle)]=optimized_params
                save_calibration_params(optimized_params, str(photo)+"_"+str(angle))

    csv_data = [{"id": key, "rotx": value[0], "roty": value[1], "rotz": value[2], "tx": value[3], "ty": value[4], "tz": value[5]} for key, value in array_calibration.items()]
    # Define CSV file name to save  all calibrations
    csv_file = f'calibrations{id}.csv'
    # Write to CSV file
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["id", "rotx", "roty", "rotz", "tx", "ty", "tz"])
        writer.writeheader()
        writer.writerows(csv_data)

generate_all_calibrations(initial_params,bnds,0.001,"01")                                                             

def compute_results(data):
    array_result={}
    for d in data:
        if not d["valid"]:
            continue
        id=d["id"]
        print(d["id"])
        file_name = id
        optimized_params = load_calibration_params(file_name)
        optimized_R = rotation_matrix_from_params(optimized_params[:3])
        optimized_t = optimized_params[3:]
        image_width = d["image_width"]
        image_height = d["image_height"]
        p1_top,p2_top,residual_distance_normalized = get_3d_point_cam1_2_from_coordinates(d["keypoints_top"][0], d["keypoints_top"][1], image_width, image_height, optimized_R, optimized_t, False)
        p1_bottom,p2_bottom,residual_distance_normalized = get_3d_point_cam1_2_from_coordinates(d["keypoints_bottom"][0], d["keypoints_bottom"][1], image_width, image_height, optimized_R, optimized_t)
        panneau_size_1 = np.linalg.norm(p1_top-p1_bottom)
        panneau_size_2 = np.linalg.norm(p2_top-p2_bottom)
        panneau_size=(panneau_size_1+panneau_size_2)/2
        panneau_height_1=p1_top[1]
        panneau_height_2=p2_top[1]
        panneau_height=(panneau_height_1+panneau_height_2)/2
        array_result[id]=[panneau_size,panneau_height_1]
        print(f'panneau size {np.linalg.norm(panneau_size)}')
        print(f'panneau height {panneau_height_1} {panneau_height_2} {panneau_height}')
        print(f'p1_top {p1_top}')
        print(f'{d["distances"][0]} vs {np.linalg.norm(p1_top)}, {d["distances"][1]} vs {np.linalg.norm(p2_top)}')
        print("\n")
    
    csv_data = [{"id": key, "taille": value[0], "hauteur": value[1]} for key, value in array_result.items()]
    csv_file = f'resultats.csv'
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["id", "taille", "hauteur"])
        writer.writeheader()
        writer.writerows(csv_data)

compute_results(data)



    
