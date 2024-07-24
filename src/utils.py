import cv2
from src.calibrate_Cube import get_cube_front

def load_and_preprocess_cube_front_images():
    cube_images_paths = []
    for index in range(5):
        for sub in range(3):
            if index == 2 and sub==2:
                continue
            cube_images_paths.append(f'/Users/michaelargi/projects/panneaux/stereo_cam/auto_cube/D_P{index+1}_CAM_G_{sub}_CUBE.png')
            cube_images_paths.append(f'/Users/michaelargi/projects/panneaux/stereo_cam/auto_cube/D_P{index+1}_CAM_D_{sub}_CUBE.png')
    print(cube_images_paths)
    cube_images = [cv2.imread(img_path) for img_path in cube_images_paths]
    # Assuming get_cube_front is a function that processes each image
    images = [get_cube_front(img) for img in cube_images]
    return images