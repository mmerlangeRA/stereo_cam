import sys

import os

def set_paths():
    current_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    print("project_root:", project_root)
    sys.path.insert(0, project_root)

def get_image_folder_path():
    current_dir = os.path.dirname(__file__)
    return os.path.join(current_dir, 'images')
