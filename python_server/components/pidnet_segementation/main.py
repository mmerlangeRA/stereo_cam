import cv2

from python_server.utils.path_helper import get_uploaded_photos_path, get_processed_path, get_public_processed_path
from src.pidnet.main import segment_image

def segment_image_from_name(image_name:str)->str:
    image_path = get_uploaded_photos_path(image_name)
    print(image_path)
    try:
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        sv_img,pred = segment_image(img)
        print(sv_img.shape)
        processed_path = get_processed_path(image_name)
        print(processed_path)
        cv2.imwrite(processed_path, sv_img)
        #sv_img.save(processed_path)
        public_path = get_public_processed_path(image_name)
        return public_path
    except FileNotFoundError as e:
        raise FileNotFoundError("Image not found ", e)
    except Exception as e :
        raise SystemError(e)
    


