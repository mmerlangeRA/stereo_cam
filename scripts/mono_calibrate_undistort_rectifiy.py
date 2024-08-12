import cv2
import os
from bootstrap import set_paths
set_paths()
from src.calibration.stereo_rectify import rectify_images
from src.utils.path_utils import find_images_paths_in_folder, load_and_preprocess_cube_front_images
from src.calibration.cube import compute_cube_calibration, read_calibration, save_calibration, undistort_and_crop

# Chessboard' size = nb of inner corners
chessboard_size = (9,6) 
# physical size of each square
square_size = 0.025

#Path to calibration matrix
mono_calibration_path=r"C:\Users\mmerl\projects\stereo_cam\calibration\mono_calibration_matrix.yaml"
mono_calibration_cube_photo_folder = r"C:\Users\mmerl\projects\stereo_cam\Calibrate_CUBE"
stereo_images_folder= r'C:\Users\mmerl\projects\stereo_cam\Photos'

ouput_rectified_width=664

folder_name = "undistorted_CUBE"
folder_path_for_undistorted = os.path.join(os.getcwd(),folder_name)
folder_path_for_undistorted= r"C:\Users\mmerl\projects\stereo_cam\undistorted_CUBE"

if not os.path.exists(folder_path_for_undistorted):
    os.makedirs(folder_path_for_undistorted)

#get or compute calibration matrix or one camera
if os.path.exists(mono_calibration_path):
    mtx, dist,rmse = read_calibration(mono_calibration_path)
else:
    image_paths = find_images_paths_in_folder(mono_calibration_cube_photo_folder)
    mtx, dist,ret = compute_cube_calibration(image_paths,chessboard_size,square_size)
    save_calibration(mtx,dist,ret, mono_calibration_path)

print(mtx)
print(dist)


# Undistort and rectify
front_images =load_and_preprocess_cube_front_images(stereo_images_folder,True)

nb_pairs = int(len(front_images)/2)

for i in range(nb_pairs):
    leftImg = front_images[2*i]
    rightImg = front_images[2*i+1]
    undistorted_left,newcameramtx = undistort_and_crop(leftImg, mtx, dist)
    undistorted_right,newcameramtx = undistort_and_crop(rightImg, mtx, dist)
    output_path = os.path.join(folder_path_for_undistorted, f'{i}_left.png')
    cv2.imwrite(output_path, undistorted_left)
    output_path = os.path.join(folder_path_for_undistorted, f'{i}_right.png')
    cv2.imwrite(output_path, undistorted_right)
    h,w=undistorted_left.shape[:2]
    ouput_rectified_height=int((h/w)*ouput_rectified_width)
    undistorted_left = cv2.resize(undistorted_left, (ouput_rectified_width, ouput_rectified_height))
    undistorted_right = cv2.resize(undistorted_right, (ouput_rectified_width, ouput_rectified_height))
    imgLeft_rectified,imgRight_rectified = rectify_images(undistorted_left,undistorted_right)
    cv2.imwrite(os.path.join(folder_path_for_undistorted,f'{i}_rectified_left.jpg'), imgLeft_rectified)
    cv2.imwrite(os.path.join(folder_path_for_undistorted,f'{i}_rectified_right.jpg'), imgRight_rectified)

print(newcameramtx)
#now get calibration with und

""" for i in range(0,len(images),2):
    focal_length,cost,_,_ =compute_auto_calibration_for_images(images[i:i+2])
    costs.append(cost)
    focal_lengths.append(focal_length) """


    
    