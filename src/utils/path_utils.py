import os


def find_image_path(folder, image_name):
    for root, dirs, files in os.walk(folder):
        if image_name in files:
            return os.path.join(root, image_name)
    return None

def create_static_folder():
    static_folder = os.path.join(os.getcwd(), 'static')
    if not os.path.exists(static_folder):
        os.makedirs(static_folder)
    return static_folder

def get_static_path(filename):
    static_folder = create_static_folder()
    return os.path.join(static_folder, filename)

def get_tmp_static_folder():
    tmp_folder = os.path.join(create_static_folder(), 'tmp')
    if not os.path.exists(tmp_folder):
        os.makedirs(tmp_folder)
    return tmp_folder