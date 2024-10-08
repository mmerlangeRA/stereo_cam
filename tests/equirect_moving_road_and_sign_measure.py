import time
initialization_start_time = time.time()

from bootstrap import set_paths
set_paths()

import csv
import cv2
import numpy as np
from src.road_detection.RoadSegmentator import SegFormerRoadSegmentator
from src.road_detection.common import AttentionWindow
from src.utils.path_utils import get_ouput_path
from src.calibration.equirectangular.main import auto_compute_cam2_transform
from src.triangulate.main import get_3d_point_cam1_2_from_coordinates
from src.utils.TransformClass import Transform, TransformBounds
from src.road_detection.StereoEquirectStereoDetector import EquirectStereoRoadDetector


frames =[
        {
        "frame_id": 1,
        "imgLeft" :r"C:\Users\mmerl\projects\stereo_cam\data\Photos\P5\D_P5_CAM_G_2_EAC.png",
        "imgRight":r"C:\Users\mmerl\projects\stereo_cam\data\Photos\P5\D_P5_CAM_D_2_EAC.png",
        "keypoints_camL": [ 
             [3034, 1258],
            [3100.0, 1258.0],
            [3100.0,1328.0],
            [3034.0, 1329.0]
        ],
        "keypoints_camR": [
             [2851, 1268],
             [2926.0,1265.0],
             [2926.0, 1343.0],
             [2853, 1342]
        ]
    },
    {
        "frame_id": 21,
        "imgLeft":r"C:\Users\mmerl\projects\stereo_cam\data\Photos\test\image_L_21_20240730_143131_519000_2301.jpg",
        "imgRight":r"C:\Users\mmerl\projects\stereo_cam\data\Photos\test\image_R_21_20240730_143124_315000_2014.jpg",
        "keypoints_camL": [ 
             [3360.0,1275.0],
            [3423.0, 1275.0],
            [3423.0,1338.0],
            [3360.0, 1338.0]
        ],
        "keypoints_camR": [
             [3416.0, 1278.0],
             [3476.0,1278.0],
             [3476.0, 1338.0],
             [3416.0,1338.0]
        ]
    },
    {
        "frame_id": 65,
        "imgLeft":r"C:\Users\mmerl\projects\stereo_cam\data\Photos\test\image_L_65_20240730_143131_519000_3022.jpg",
        "imgRight":r"C:\Users\mmerl\projects\stereo_cam\data\Photos\test\image_R_65_20240730_143124_315000_2591.jpg",
        "keypoints_camL": [ 
            [3050.0,1485.0],
            [3131.0, 1485.0],
            [3131.0,1573.0],
            [3050.0, 1573.0]
        ],
        "keypoints_camR": [
             [3209.0, 1463.0],
             [3270.0,1460.0],
             [3270.0, 1538.0],
             [3209.0,1538.0]
        ]
    },   
    {
        "frame_id": 81,
        "imgLeft":r"C:\Users\mmerl\projects\stereo_cam\data\Photos\test\image_L_81_20240730_143131_519000_3250.jpg",
        "imgRight":r"C:\Users\mmerl\projects\stereo_cam\data\Photos\test\image_R_81_20240730_143124_315000_2773.jpg",
        "keypoints_camL": [ 
            [250.0,1444.0],
            [314.0, 1444.0],
            [314.0, 1511.0],
            [250.0,1511.0],
        ],
        "keypoints_camR": [
             [138.0, 1453.0],
             [208.0,1453.0],
             [208.0,1521.0],
             [138.0, 1521.0]
        ]
    },
    
    {
        "frame_id": 294,
        "imgLeft":r"C:\Users\mmerl\projects\stereo_cam\data\Photos\test\image_L_294_20240730_143131_519000_6899.jpg",
        "imgRight":r"C:\Users\mmerl\projects\stereo_cam\data\Photos\test\image_R_294_20240730_143124_315000_5692.jpg",
        "keypoints_camL": [ 
             [3208.0,1246.0],
            [3265.0, 1246.0],
            [3265.0, 1316.0],
            [3208.0,1316.0],
        ],
        "keypoints_camR": [
             [3300.0, 1254.0],
             [3352.0,1254.0],
             [3352.0,1318.0],
             [3300.0, 1318.0],
        ]
    },
     {
        "frame_id": 303,
        "imgLeft": r"C:\Users\mmerl\projects\stereo_cam\data\Photos\test\image_L_303_20240730_143131_519000_7086.jpg", 
        "imgRight" :r"C:\Users\mmerl\projects\stereo_cam\data\Photos\test\image_R_303_20240730_143124_315000_5842.jpg",
        "keypoints_camL": [
            [ 2983.0, 1412.0],
            [3056.0, 1412.0],
            [3056.0, 1492.0],
             [ 2983.0, 1492.0]
        ],
        "keypoints_camR": [
            [3098.0,1402.0],
            [3165.0,1402.0],
            [3165.0,1478.0],
            [3098.0,1478.0],
        ],
                },

    {
        "frame_id": 307,
        "imgLeft": r"C:\Users\mmerl\projects\stereo_cam\data\Photos\307\image_L_307_20240730_143131_519000_7147.jpg", 
        "imgRight" :r"C:\Users\mmerl\projects\stereo_cam\data\Photos\307\image_R_307_20240730_143124_315000_5891.jpg",
        "keypoints_camL": [
            [3622.0,1262.0],
            [3668.0,1277.0],
            [3668.0,1404.0],
            [3622.0,1404.0],
        ],
        "keypoints_camR": [
            [3698.0, 1285.0],
            [3741.0,1296.0],
            [3741.0, 1392.0],
            [3698.0, 1392.0],
        ]
    },
]

#frames=[frames[1]]

image_width =5376 
image_height= 2688 

#calibration parameters
optimize_global= True
inlier_threshold = 0.001
base_line=1.125
angle_max = np.pi*5./180.
dt_max_y = 0.05
dt_max_z= 0.7

estimated_camRight_transform = Transform(xc=1.1100000000010084, yc=-0.015367638222386357, zc=0.026834207520040555, pitch=0.023162473327744338, yaw=0.07609111219036904, roll=0.009961248317160167)
estimated_camRight_transform.scale_translation_from_x(baseline=base_line)
transformBounds= TransformBounds(baseline=base_line, dt_max_y=dt_max_y,dt_max_z=dt_max_z, angle_max=angle_max)

#road detection parameters
window_left=0.4
window_right = 0.6
road_window_top = 0.53
window_top = 0.4
window_bottom = 0.6
road_debug = True
camHeight = 1.65
degree= 2

# Attention window for segementation and road detection
limit_left = int(window_left * image_width)
limit_right = int(window_right * image_width)
limit_top = int(window_top * image_height)
limit_bottom = int(window_bottom * image_height)
window = AttentionWindow(limit_left, limit_right, limit_top, limit_bottom)


roadSegmentator = SegFormerRoadSegmentator(kernel_width=20, use_1024=False, debug=road_debug)
roadDetector = EquirectStereoRoadDetector(roadSegmentator=roadSegmentator,
                                          window=window,road_down_y=camHeight, 
                                          degree=degree, road_contour_top=road_window_top,
                                          estimated_cam2_transform=estimated_camRight_transform,
                                        debug=road_debug)
initialization_end_time = time.time()

print("Time taken for initialization: ", initialization_end_time - initialization_start_time, "seconds")

verbose = True 

computed=[]

for frame in frames:
    frameId=frame["frame_id"]
    
    invert_left_right = frame["keypoints_camL"][0][0]<frame["keypoints_camR"][0][0]

    print(frameId,invert_left_right)

    
    name_left_kps = "keypoints_camR" if invert_left_right else "keypoints_camL"
    name_right_kps = "keypoints_camL" if invert_left_right else "keypoints_camR"

    name_left_img = "imgRight" if invert_left_right else "imgLeft"
    name_right_img = "imgLeft" if invert_left_right else "imgRight"

    left_image = cv2.imread(frame[name_left_img] )
    right_image = cv2.imread(frame[name_right_img])

    keypoints_cam1_TL=frame[name_left_kps][0]
    keypoints_cam2_TL =frame[name_right_kps][0]
    keypoints_cam1_BL=frame[name_left_kps][3]
    keypoints_cam2_BL =frame[name_right_kps][3]

    if optimize_global:           
        refined_camRight_transform,ratio = auto_compute_cam2_transform(left_image, right_image,estimatedTransform= estimated_camRight_transform, 
                                                   transformBounds=transformBounds,inlier_threshold=inlier_threshold,verbose=True)
        refined_camRight_transform.scale_translation_from_x(baseline=base_line)
        print("refined best")
        print(ratio,refined_camRight_transform)

    nb_kps = len(frame[name_left_kps])

    topLeft1,topLeft2,residual_in_m1 = get_3d_point_cam1_2_from_coordinates(
        tuple(keypoints_cam1_TL), 
        tuple(keypoints_cam2_TL), image_width, image_height, refined_camRight_transform.rotationMatrix,refined_camRight_transform.translationVector, verbose)

    bottomLeft1,bottomLeft2,residual_in_m2 = get_3d_point_cam1_2_from_coordinates(
        tuple(keypoints_cam1_BL), 
        tuple(keypoints_cam2_BL), image_width, image_height, refined_camRight_transform.rotationMatrix,refined_camRight_transform.translationVector, verbose)

    sign_width1 = np.linalg.norm(np.array(bottomLeft1) - np.array(topLeft1))
    sign_width2 = np.linalg.norm(np.array(bottomLeft2) - np.array(topLeft2))

    if verbose:
        print(f"topLeft 3D Point Camera 1: {topLeft1}")
        print(f"topLeft 3D Point Camera 2: {topLeft2}")
        print(f"Residual: {residual_in_m1}")
        print(f"bottomLeft 3D Point Camera 1: {bottomLeft1}")
        print(f"bottomLeft 3D Point Camera 2: {bottomLeft2}")
        print(f"Residual: {residual_in_m2}")
        print(f"{frameId} width cam1 {sign_width1}")
        print(f"width cam2 {sign_width2}")

    #now road size
    start_time = time.time()
    roadDetector.estimated_cam2_transform = refined_camRight_transform
    concatenated_horizontal_img = np.concatenate((left_image, right_image), axis=1)
    road_width = roadDetector.compute_road_width(concatenated_horizontal_img)
    end_time = time.time()
    if verbose:
        cv2.imwrite(get_ouput_path(f'{frameId}_road_window_left.png'), window.crop_image(left_image))
        cv2.imwrite(get_ouput_path(f'{frameId}_road_window_right.png'), window.crop_image(right_image))
        if road_width <0:
            print("No road detected")
        else:
            print(f'road_width {road_width} in {end_time- start_time}')
            road_vector = roadDetector.road_vector
            left_cam_transform = Transform()
            debug_img =roadDetector._debug_display_projected_road_on_image(left_image,roadDetector.left_img_contour_left, roadDetector.left_img_contour_right,left_cam_transform,road_vector)
            cv2.imwrite(get_ouput_path(f'{frameId}_road_debug_left.png'), debug_img)
            debug_img =roadDetector._debug_display_projected_road_on_image(right_image,roadDetector.right_img_contour_left, roadDetector.right_img_contour_right ,refined_camRight_transform,road_vector)
            cv2.imwrite(get_ouput_path(f'{frameId}_road_debug_right.png'), debug_img)

    computed.append({
        "frame_id":frameId,
        "sign_width1":round(sign_width1,2),
        "sign_width2":round(sign_width2,2),
        "dx":round(topLeft1[0],2),
        "dy":round(topLeft1[1],2),
        "dz":round(topLeft1[2],2),
        "residual_in_m1":residual_in_m1,
        "residual_in_m2":residual_in_m2,
        "road_width":road_width
    })


if computed:
    headers = computed[0].keys()
else:
    headers = []

save_path = r"C:\Users\mmerl\projects\stereo_cam\data\Photos\test\output.csv"
with open(save_path, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=headers)
    writer.writeheader()
    for data in computed:
        writer.writerow(data)






