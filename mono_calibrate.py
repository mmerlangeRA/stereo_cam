import numpy as np
import cv2 as cv
import glob
import yaml
import os

from src.calibrate import compute_and_save_calibration, get_cube_subs, read_calibration, undistort

# Termination criteria


chessboard_size = (9,6) 
# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)



calibration_path="calibration_matrix.yaml"

is_cube=True
if is_cube:
    images = glob.glob('Calibrate_CUBE/*.png')
else:
    images = glob.glob('Calibrate_EAC/*.png')

if os.path.exists(calibration_path):
    mtx, dist,rmse = read_calibration(calibration_path)
else:
    mtx, dist = compute_and_save_calibration(images,chessboard_size,is_cube)

print(mtx)
print(dist)

folder_name = "undistorted_CUBE"

folder_path = os.path.join(os.getcwd(),folder_name)

for fname in images:
    base_name = os.path.basename(fname)
    print(f'undistorting {base_name}')
    if is_cube:
        img = cv.imread(fname)
        """         
        undistorted = undistort(img, mtx, dist)
        output_path = os.path.join(folder_path, f'{base_name}')
        cv.imwrite(output_path, undistorted)
        continue 
        """
        sub_images=get_cube_subs(img)
        index=0
        for sub in sub_images:
            index+=1
            undistorted = undistort(sub, mtx, dist)
            output_path = os.path.join(folder_path, f'{index}_{base_name}')
            cv.imwrite(output_path, undistorted)


    
    #undistorted = undistort_and_crop(cv.imread(fname), mtx, dist)
    #cv.imwrite(f'undistorted_and_cropped{base_name}', undistorted)

'''
# Undistort an image
img = cv.imread('Calibrate_EAC/GS010064_000490.png')
h, w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

# Undistort
dst = cv.undistort(img, mtx, dist, None, newcameramtx)
cv.imwrite('calibresult_no_cropping.png', dst)
cv.imshow('dst', dst)
cv.waitKey(0)
# Crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('calibresult.png', dst)
cv.imshow('dst', dst)
cv.waitKey(0)
cv.destroyAllWindows()
'''

