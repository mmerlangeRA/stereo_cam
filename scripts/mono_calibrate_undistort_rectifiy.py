import cv2
import os
from bootstrap import set_paths
set_paths()
from src.calibration.StereoCalibrator import StereoCalibrator
from src.utils.path_utils import find_images_paths_in_folder, load_and_preprocess_cube_front_images

# Chessboard' size = nb of inner corners
chessboard_size = (9,6) 
# physical size of each square
square_size = 0.025

calibrator = StereoCalibrator(verbose=True, estimated_base_line_in_m=1.12,calibration_file_name="calibrator_matrix.json")

calibrator.reset()


#Path to calibration matrix
calibration_path=calibrator.calibration_file_path
mono_calibration_cube_photo_folder = r"C:\Users\mmerl\projects\stereo_cam\Calibrate_CUBE"
stereo_images_folder= r'C:\Users\mmerl\projects\stereo_cam\Photos'


folder_name = "undistorted_CUBE"
folder_path_for_undistorted = os.path.join(os.getcwd(),folder_name)
folder_path_for_undistorted= r"C:\Users\mmerl\projects\stereo_cam\undistorted_CUBE"

if not os.path.exists(folder_path_for_undistorted):
    os.makedirs(folder_path_for_undistorted)

#get or compute calibration matrix or one camera
if os.path.exists(calibration_path):
    calibrator.read_calibration()
    calibrator.compute_stereo_rectified_Z0()
else:
    image_paths = find_images_paths_in_folder(mono_calibration_cube_photo_folder)
    mtx, dist,ret = calibrator.compute_mono_chessboard_calibration(image_paths,chessboard_size,square_size)
    calibrator.save_calibration()

print(calibrator.calibration.mono_K)
print(calibrator.calibration.mono_ret)


# Undistort and rectify
front_images =load_and_preprocess_cube_front_images(stereo_images_folder,True)

nb_pairs = int(len(front_images)/2)

left_img_paths = []
right_img_paths = []
for i in range(nb_pairs):
    leftImg = front_images[2*i]
    rightImg = front_images[2*i+1]

    undistorted_left,newcameramtx = calibrator.undistort_and_crop(leftImg)
    undistorted_right,newcameramtx = calibrator.undistort_and_crop(rightImg)
    undistorted_left_path = os.path.join(folder_path_for_undistorted, f'{i}_left.png')
    undistorted_right_path = os.path.join(folder_path_for_undistorted, f'{i}_right.png')

    cv2.imwrite(os.path.join(folder_path_for_undistorted, f'{i}_left.png'), undistorted_left)
    cv2.imwrite(os.path.join(folder_path_for_undistorted, f'{i}_right.png'), undistorted_right)
    left_img_paths.append(undistorted_left_path)
    right_img_paths.append(undistorted_right_path)
calibrator.compute_global_auto_calibration_undistorted(left_img_paths, right_img_paths)
calibrator.save_calibration()

rectified_left_img_paths = []
rectified_right_img_paths = []
for i in range(nb_pairs):
    undistorted_left = cv2.imread(left_img_paths[i])
    undistorted_right = cv2.imread(right_img_paths[i])

    imgLeft_rectified,imgRight_rectified = calibrator.rectify_undistorted_images(undistorted_left,undistorted_right)
    rectified_left_path = os.path.join(folder_path_for_undistorted, f'{i}_rectified_left.jpg')
    rectified_right_path = os.path.join(folder_path_for_undistorted, f'{i}_rectified_right.jpg')
    rectified_left_img_paths.append(rectified_left_path)
    rectified_right_img_paths.append(rectified_right_path)
    cv2.imwrite(rectified_left_path, imgLeft_rectified)
    cv2.imwrite(rectified_right_path, imgRight_rectified)

calibrator.compute_global_auto_calibration_rectified(rectified_left_img_paths, rectified_right_img_paths)
calibrator.compute_stereo_rectified_Z0()
calibrator.save_calibration()
    


#now get calibration with und

""" for i in range(0,len(images),2):
    focal_length,cost,_,_ =compute_auto_calibration_for_images(images[i:i+2])
    costs.append(cost)
    focal_lengths.append(focal_length) """


    
    