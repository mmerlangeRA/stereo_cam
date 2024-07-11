import os
from python_server.settings.settings import settings

def find_image_path(folder, image_name)->str:
    for root, dirs, files in os.walk(folder):
        if image_name in files:
            return os.path.join(root, image_name)
    return None

def create_static_folder(sub_folder_name="")->str:
    static_folder = os.path.join(os.getcwd(), 'static')
    if not os.path.exists(static_folder):
        os.makedirs(static_folder)
    if sub_folder_name != "":
        static_folder = os.path.join(static_folder, sub_folder_name)
        if not os.path.exists(static_folder):
            os.makedirs(static_folder)
    return static_folder

def get_static_path(filename="")->str:
    static_folder = create_static_folder()
    return os.path.join(static_folder, filename)

def get_public_path(filename="")->str:
    return os.path.join(settings().server.base_url,filename)

def get_photos_path(filename)->str:
    photo_static_folder = create_static_folder(settings().data.photo_data_folder)
    return os.path.join(photo_static_folder, filename)

def get_processed_path(filename)->str:
    processed_static_folder = create_static_folder(settings().data.processed_data_folder)
    return os.path.join(processed_static_folder, filename)

def get_public_photo_path(filename)->str:
    photo_folder = settings().data.photo_data_folder
    create_static_folder(photo_folder)
    return settings().server.base_url +"/static/"+photo_folder+"/"+filename

def get_public_processed_path(filename)->str:
    processed_folder = settings().data.processed_data_folder
    create_static_folder(processed_folder)
    return settings().server.base_url +"/static/"+processed_folder+"/"+filename

def get_tmp_static_folder()->str:
    tmp_folder = os.path.join(create_static_folder(), 'tmp')
    if not os.path.exists(tmp_folder):
        os.makedirs(tmp_folder)
    return tmp_folder