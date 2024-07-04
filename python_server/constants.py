from pathlib import Path

import os

from src.utils.path_utils import create_static_folder,get_tmp_static_folder

PROJECT_ROOT_PATH: Path = Path(__file__).parents[1]

PHOTO_FOLDER_PATH = os.path.join(PROJECT_ROOT_PATH, "Photos")

STATIC_FOLDER_PATH = create_static_folder()

TMP_FOLDER_PATH = get_tmp_static_folder()