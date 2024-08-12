import os
from typing import List
import cv2
from src.utils.cube_image import get_cube_front_image

def get_root() -> str:
    #return os.getcwd() 
    return r'C:\Users\mmerl\projects\stereo_cam'

def get_pretrained_model_path(lib:str,model:str) -> str:
    return os.path.join(get_root(), "pretrained_models",lib,model)

def get_static_folder_path(path="") -> str:
    return os.path.join(get_root(), "static",path)

def find_images_paths_in_folder(root_dir, extensions=[".png",".jpg",".jpeg"])->List[str]: 
    image_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            for extension in extensions:
                if file.endswith(extension):
                    image_files.append(os.path.join(root, file))
    return image_files

def load_and_preprocess_cube_front_images(folder:str,verbose=False)->List[cv2.typing.MatLike]:
    if verbose:
        print('load_and_preprocess_cube_front_images')
    cube_images_paths = []
    for index in range(5):
        subfolder = os.path.join(folder, f'P{index+1}')
        for sub in range(3):
            left_image_path = os.path.join(subfolder, f'D_P{index+1}_CAM_G_{sub}_CUBE.png')
            right_image_path = os.path.join(subfolder, f'D_P{index+1}_CAM_D_{sub}_CUBE.png')
            if os.path.exists(left_image_path) and os.path.exists(right_image_path):
                cube_images_paths.append(left_image_path)
                cube_images_paths.append(right_image_path)
    if verbose:
        print(cube_images_paths)
    cube_images = [cv2.imread(img_path) for img_path in cube_images_paths]

    images = [get_cube_front_image(img) for img in cube_images]
    return images