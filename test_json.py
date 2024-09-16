import csv
import numpy as np
from python_server.components.triangulation_equipolar.main import AutoCalibrationRequest, TriangulationRequest, auto_calibrate_equipoloar, triangulate_equipolar_points
from src.calibration.eac import getCalibrationFrom3Matching


frames =[
    {
        "frame_id": 21,
        "imgLeft":r"C:\Users\mmerl\projects\stereo_cam\Photos\test\image_L_21_20240730_143131_519000_2301.jpg",
        "imgRight":r"C:\Users\mmerl\projects\stereo_cam\Photos\test\image_R_21_20240730_143124_315000_2014.jpg",
        "keypoints_camL": [ 
             [3357.0,1272.0],
            [3426.0, 1272.0],
            [3357.0,1341.0],
            [3426.0, 1341.0]
        ],
        "keypoints_camR": [
             [3414.0, 1276.0],
             [3477.0,1276.0],
             [3414.0, 1341.0],
             [3477.0,1341.0]
        ]
    },
    {
        "frame_id": 65,
        "imgLeft":r"C:\Users\mmerl\projects\stereo_cam\Photos\test\image_L_65_20240730_143131_519000_3022.jpg",
        "imgRight":r"C:\Users\mmerl\projects\stereo_cam\Photos\test\image_R_65_20240730_143124_315000_2591.jpg",
        "keypoints_camL": [ 
            [3047.0,1482.0],
            [3133.0, 1482.0],
            [3047.0,1576.0],
            [3133.0, 1576.0]
        ],
        "keypoints_camR": [
             [3206.0, 1460.0],
             [3274.0,1460.0],
             [3206.0, 1542.0],
             [3274.0,1542.0]
        ]
    },   
    {
        "frame_id": 81,
        "imgLeft":r"C:\Users\mmerl\projects\stereo_cam\Photos\test\image_L_81_20240730_143131_519000_3250.jpg",
        "imgRight":r"C:\Users\mmerl\projects\stereo_cam\Photos\test\image_R_81_20240730_143124_315000_2773.jpg",
        "keypoints_camL": [ 
            [250.0,1444.0],
            [314.0, 1444.0],
            [250.0,1511.0],
            [314.0, 1511.0]
        ],
        "keypoints_camR": [
             [138.0, 1453.0],
             [208.0,1453.0],
             [138.0, 1521.0],
             [208.0,1521.0]
        ]
    },
    
    {
        "frame_id": 294,
        "imgLeft":r"C:\Users\mmerl\projects\stereo_cam\Photos\test\image_L_294_20240730_143131_519000_6899.jpg",
        "imgRight":r"C:\Users\mmerl\projects\stereo_cam\Photos\test\image_R_294_20240730_143124_315000_5692.jpg",
        "keypoints_camL": [ 
             [3208.0,1246.0],
            [3265.0, 1246.0],
            [3208.0,1316.0],
            [3265.0, 1316.0]
        ],
        "keypoints_camR": [
             [3300.0, 1254.0],
             [3352.0,1254.0],
             [3300.0, 1318.0],
             [3352.0,1318.0]
        ]
    },
     {
        "frame_id": 303,
        "imgLeft": r"C:\Users\mmerl\projects\stereo_cam\Photos\test\image_L_303_20240730_143131_519000_7086.jpg", 
        "imgRight" :r"C:\Users\mmerl\projects\stereo_cam\Photos\test\image_R_303_20240730_143124_315000_5842.jpg",
        "keypoints_camL": [
            [ 2983.0, 1412.0],
            [3056.0, 1412.0],
             [ 2983.0, 1492.0],
            [3056.0, 1492.0]
        ],
        "keypoints_camR": [
            [3098.0,1402.0],
            [3165.0,1402.0],
             [3098.0,1478.0],
            [3165.0,1478.0]
        ],
                },

    {
        "frame_id": 307,
        "imgLeft": r"C:\Users\mmerl\projects\stereo_cam\Photos\307\image_L_307_20240730_143131_519000_7147.jpg", 
        "imgRight" :r"C:\Users\mmerl\projects\stereo_cam\Photos\307\image_R_307_20240730_143124_315000_5891.jpg",
        "keypoints_camL": [
            [3622.0,1262.0],
            [3668.0,1277.0],
            [3622.0,1404.0],
        ],
        "keypoints_camR": [
            [3698.0, 1285.0],
            [3741.0,1296.0],
            [3698.0, 1392.0],
        ]
    },
]


image_width =5376 
image_height= 2688 

invert_left_right = True

optimize_global= False

initial_params =[0, 0, 0, 1.12, 0, 0]
bnds =[[-0.17, 0.17], [-0.17, 0.17], [-0.17, 0.17], [1.11, 1.13], [-0.12001, 0.12001], [-0.12001, 0.12001]]
inlier_threshold = 0.01

verbose = False 

computed=[]

for frame in frames:
    frameId=frame["frame_id"]
    if invert_left_right:
        imgLeft_name = frame["imgRight"]
        imgRight_name = frame["imgLeft"]
    else:
        imgLeft_name = frame["imgLeft"]
        imgRight_name = frame["imgRight"]


    if optimize_global:   
        request = AutoCalibrationRequest(
                    imgLeft_name=imgLeft_name,
                    imgRight_name=imgRight_name,
                    initial_params=initial_params,
                    bnds=bnds,
                    inlier_threshold=inlier_threshold
                )

        optimized_params_global = auto_calibrate_equipoloar(request,verbose=True)
        print(optimized_params_global)

    name_left_kps = "keypoints_camR" if invert_left_right else "keypoints_camL"
    name_right_kps = "keypoints_camL" if invert_left_right else "keypoints_camR"

    keypoints_cam1_TL=frame[name_left_kps][0]
    keypoints_cam2_TL =frame[name_right_kps][0]
    keypoints_cam1_BL=frame[name_left_kps][2]
    keypoints_cam2_BL =frame[name_right_kps][2]

    nb_kps = len(frame[name_left_kps])
    if nb_kps>4:
        sub_uv1=[]
        sub_uv2=[]
        for i in range(nb_kps):
            sub_uv1.append(frame[name_left_kps][i])
            sub_uv2.append(frame[name_right_kps][i])
            optimized_params_local,total_residual_in_m = getCalibrationFrom3Matching(sub_uv1, sub_uv2, initial_params, image_width=image_width, image_height=image_height,bnds=bnds)
        print(optimized_params_local)
        print(total_residual_in_m)

    #print(optimized_params_local)
    #R= optimized_params_local[:3]
    #t = optimized_params_local[3:6]
    R=[0.,0.,0.]
    t=[1.12,0.,0.]

    request = TriangulationRequest(
            keypoints_cam1=tuple(keypoints_cam1_TL),
            keypoints_cam2=tuple(keypoints_cam2_TL),
            image_width=image_width,
            image_height=image_height,
            R=R,
            t=t,
            verbose=verbose
        )

    topLeft1, topLeft2, residual_in_m1 = triangulate_equipolar_points(request,verbose=verbose)
    if verbose:
        print(f"3D Point Camera 1: {topLeft1}")
        print(f"3D Point Camera 2: {topLeft2}")
        print(f"Residual: {residual_in_m1}")


    request = TriangulationRequest(
            keypoints_cam1=tuple(keypoints_cam1_BL),
            keypoints_cam2=tuple(keypoints_cam2_BL),
            image_width=image_width,
            image_height=image_height,
            R=R,
            t=t,
            verbose=verbose
        )
    bottomLeft1, bottomLeft2, residual_in_m2 = triangulate_equipolar_points(request,verbose=verbose)
    
    if verbose:
        print(f"3D Point Camera 1: {bottomLeft1}")
        print(f"3D Point Camera 2: {bottomLeft2}")
        print(f"Residual: {residual_in_m2}")


    width1 = np.linalg.norm(np.array(bottomLeft1) - np.array(topLeft1))
    print(f"{frameId} width cam1 {width1}")

    width2 = np.linalg.norm(np.array(bottomLeft2) - np.array(topLeft2))
    print(f"width cam2 {width2}")

    computed.append({
        "frame_id":frameId,
        "width1":round(width1,2),
        "width2":round(width2,2),
        "dx":round(topLeft1[0],2),
        "dy":round(topLeft1[1],2),
        "dz":round(topLeft1[2],2),
        "residual_in_m1":residual_in_m1,
        "residual_in_m2":residual_in_m2
    })


if computed:
    headers = computed[0].keys()
else:
    headers = []

save_path = r"C:\Users\mmerl\projects\stereo_cam\Photos\test\output.csv"
with open(save_path, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=headers)
    writer.writeheader()
    for data in computed:
        writer.writerow(data)






