import cv2

from python_server.utils.path_helper import get_photos_path, get_processed_path, get_public_processed_path
from src.PIDNet.main import segment_image

def segment_image_from_name(image_name:str)->str:
    image_path = get_photos_path(image_name)
    print(image_path)
    try:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        sv_img = segment_image(img)
        processed_path = get_processed_path(image_name)
        sv_img.save(processed_path)
        public_path = get_public_processed_path(image_name)
        return public_path
    except FileNotFoundError as e:
        raise FileNotFoundError("Image not found ", e)
    except Exception as e :
        raise SystemError(e)
    


