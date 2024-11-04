from bootstrap import set_paths
set_paths()

import csv
import cv2
import numpy as np
from src.calibration.equirectangular.main import auto_compute_cam2_transform, getRefinedTransformFromKPMatching
from src.triangulate.main import get_3d_point_cam1_2_from_coordinates
from src.utils.TransformClass import Transform, TransformBounds

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

def get_2_cam_refpoints(points):
    points = np.array(points)
    top = (points[0]+points[1])/2.
    bottom = (points[2]+points[3])/2.
    return points[0], points[3]

#frames=[frames[1]]

# Assumptions
## Image size
image_width =5376 
image_height= 2688 

invert_left_right = True

should_optimize_global= True
inlier_threshold = 0.005

## Cameras geometry
base_line=1.125
angle_max = np.pi*5./180.# max error on rotation
dt_max_y = 0.05# max error on y translation 
dt_max_z= 0.7# max error on z translation 
default_transform = Transform(base_line, 0., 0., 0., 0., 0.)
best_results = Transform(base_line,0.,0.,0.,0.,0.)

#Give best estimation, should be updated at every frame
estimated_transform = Transform(xc=1.1100000000010084, yc=-0.015367638222386357, zc=0.026834207520040555, pitch=0.023162473327744338, yaw=0.07609111219036904, roll=0.009961248317160167)
estimated_transform.scale_translation_from_x(baseline=base_line)

# Idea here: how to choose the corners to minimize the computation time. Unclear for now
transformBounds= TransformBounds(baseline=base_line, dt_max_y=dt_max_y,dt_max_z=dt_max_z, angle_max=angle_max)
top_limit=int(image_height*0.45)
bottom_limit=int(image_height*0.8)
top_limit=0
bottom_limit=image_height

verbose = True 

computed=[]

for frame in frames:
    frameId=frame["frame_id"]
    tpl1=frame["keypoints_camL"][0][0]
    tpl2 =frame["keypoints_camR"][0][0]
    invert_left_right = True if frameId>1 else False
    print(f"frame {frameId},invert_left_right {invert_left_right}")
    name_left_kps = "keypoints_camR" if invert_left_right else "keypoints_camL"
    name_right_kps = "keypoints_camL" if invert_left_right else "keypoints_camR"

    name_left_img = "imgRight" if invert_left_right else "imgLeft"
    name_right_img = "imgLeft" if invert_left_right else "imgRight"

    left_image = cv2.imread(frame[name_left_img] )
    right_image = cv2.imread(frame[name_right_img])

    keypoints_cam1=frame[name_left_kps]
    keypoints_cam2=frame[name_right_kps]

    if should_optimize_global:           
        best_results,ratio = auto_compute_cam2_transform(left_image, right_image,estimatedTransform= estimated_transform, 
                                                   transformBounds=transformBounds,inlier_threshold=inlier_threshold,verbose=True,frame_id=frameId)
        best_results.scale_translation_from_x(baseline=base_line)
        print("ratio, refined best:",ratio, best_results)


    # Idea below is to optimize localy the transform, but it is not working well
    else:
        nb_kps = len(frame[name_left_kps])
        if nb_kps>4:
            sub_uv1=[]
            sub_uv2=[]
            for i in range(nb_kps):
                sub_uv1.append(frame[name_left_kps][i])
                sub_uv2.append(frame[name_right_kps][i])
                refined_transform_local,total_residual_in_m = getRefinedTransformFromKPMatching(sub_uv1, sub_uv2, initial_params, image_width=image_width, image_height=image_height,bnds=bnds)
            print(refined_transform_local)
            print(total_residual_in_m)
    # compute sign size and position. In this example, it's on left side, should rather be in top/bottom center
    
    keypoints_1_cam1,keypoints_2_cam1=get_2_cam_refpoints(keypoints_cam1)
    keypoints_1_cam2,keypoints_2_cam2=get_2_cam_refpoints(keypoints_cam2)
    ## compute top left corner triangulation
    top_cam1,top_cam2,residual_in_m1 = get_3d_point_cam1_2_from_coordinates(
        tuple(keypoints_1_cam1), 
        tuple(keypoints_1_cam2), image_width, image_height, best_results.rotationMatrix,best_results.translationVector, verbose)

    if verbose:
        print(f"Top 3D Point Camera 1: {top_cam1}")
        print(f"Top 3D Point Camera 2: {top_cam2}")
        print(f"Residual: {residual_in_m1}")

    ## compute bottom left corner triangulation
    bottom_cam1,bottom_cam2,residual_in_m2 = get_3d_point_cam1_2_from_coordinates(
        tuple(keypoints_2_cam1), 
        tuple(keypoints_2_cam2), image_width, image_height, best_results.rotationMatrix,best_results.translationVector, verbose)

    if verbose:
        print(f"Bottom 3D Point Camera 1: {bottom_cam1}")
        print(f"Bottom 3D Point Camera 2: {bottom_cam2}")
        print(f"Residual: {residual_in_m2}")


    width1 = np.linalg.norm(np.array(bottom_cam1) - np.array(top_cam1))
    print(f"{frameId} width cam1 {width1}")

    width2 = np.linalg.norm(np.array(bottom_cam2) - np.array(top_cam2))
    print(f"width cam2 {width2}")

    computed.append({
        "frame_id":frameId,
        "ratio":round(ratio,2),
        "invert_left_right":invert_left_right,
        "width1":round(width1,2),
        "width2":round(width2,2),
        "dx":round(top_cam1[0],2),
        "dy":round(top_cam1[1],2),
        "dz":round(top_cam1[2],2),
        "residual_in_m1":round(residual_in_m1,3),
        "residual_in_m2":round(residual_in_m2,3),
        "best_results":best_results
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






