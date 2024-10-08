from bootstrap import set_paths
set_paths()
import cv2
from matplotlib import pyplot as plt
import numpy as np

from src.road_detection.RoadSegmentator import SegFormerRoadSegmentator
from src.road_detection.RoadDetector import EquirectMonoRoadDetector
from src.road_detection.common import AttentionWindow
from src.utils.curve_fitting import Road_line_params, find_best_2_polynomial_curves, vizualize_road_equirectangular
from src.utils.TransformClass import Transform
from src.utils.path_utils import get_ouput_path
from src.calibration.cube.cube import load_calibration_params
from src.calibration.equirectangular.main import compute_stereo_matched_KP
from src.triangulate.main import get_3d_point_cam1_2_from_coordinates
from src.utils.coordinate_transforms import rotation_matrix_from_vector3D

if __name__ == '__main__':
    img_left_path=r'C:\Users\mmerl\projects\stereo_cam\data\Photos\P5\D_P5_CAM_G_2_EAC.png'
    img_left = cv2.imread(img_left_path, cv2.IMREAD_COLOR)

    img_right_path=r'C:\Users\mmerl\projects\stereo_cam\data\Photos\P5\D_P5_CAM_D_2_EAC.png'
    img_right = cv2.imread(img_right_path, cv2.IMREAD_COLOR)


    def compute_xs_from_y(img_left,img_right,img_left_test_y):
        image_height, image_width = img_left.shape[:2]

        window_left=0.2
        window_right = 0.8
        window_top = 0.3
        window_bottom = 0.57
        debug = True
        camHeight = 1.65
        degree = 2

        # Attention window for segementation and road detection
        limit_left = int(window_left * image_width)
        limit_right = int(window_right * image_width)
        limit_top = int(window_top * image_height)
        limit_bottom = int(window_bottom * image_height)
        window = AttentionWindow(limit_left, limit_right, limit_top, limit_bottom)
    
        #processing
        roadSegmentator = SegFormerRoadSegmentator(kernel_width=10, use_1024=True, debug=True)
        roadDetector = EquirectMonoRoadDetector(roadSegmentator=roadSegmentator,window=window,road_down_y=camHeight, degree=degree, debug=False)
        
        cv2.imwrite(get_ouput_path( "window_left.png"),window.crop_image(img_left))
        cv2.imwrite(get_ouput_path( "window_right.png"),window.crop_image(img_right))
        
        left_countours=roadDetector._get_road_contours(img_left)
        right_countours=roadDetector._get_road_contours(img_right)
        left_contour = max(left_countours, key=cv2.contourArea)
        right_contour = max(right_countours, key=cv2.contourArea)
        
        left_contour_points = left_contour[:, 0, :]
        left_contour_x = left_contour_points[:, 0]
        left_contour_y = left_contour_points[:, 1]
        left_poly_model_left, left_poly_model_right, left_y_inliers_left, left_y_inliers_right = find_best_2_polynomial_curves(left_contour,degree=degree)
        
        right_contour_points = right_contour[:, 0, :]
        right_contour_x = right_contour_points[:, 0]
        right_contour_y = right_contour_points[:, 1]
        right_poly_model_left, right_poly_model_right, right_y_inliers_left, right_y_inliers_right = find_best_2_polynomial_curves(right_contour,degree=degree)
        
        min_y_inliers_first = np.min(left_y_inliers_left)
        min_y_inliers_second = np.min(left_y_inliers_right)
        max_y_inliers_first = np.max(left_y_inliers_left)
        max_y_inliers_second = np.max(left_y_inliers_right)

        minY = max(min_y_inliers_first, min_y_inliers_second)
        maxY = min(max_y_inliers_first, max_y_inliers_second)

        left_contour_left=[]
        left_contour_right=[]
        right_contour_left=[]
        right_contour_right=[]

        for y1 in range(minY, maxY,2):
            left_x_left = int(left_poly_model_left.predict([[y1]])[0])
            left_x_right = int(left_poly_model_right.predict([[y1]])[0])
            left_contour_left.append((left_x_left, int(y1)))
            left_contour_right.append((left_x_right, int(y1)))

            right_x_left = int(right_poly_model_left.predict([[y1]])[0])
            right_x_right = int(right_poly_model_right.predict([[y1]])[0])
            right_contour_left.append((right_x_left, int(y1)))
            right_contour_right.append((right_x_right, int(y1)))

        contour_left_img = img_left.copy()
        contour_right_img = img_right.copy()
        for p in left_contour_left:
            cv2.circle(contour_left_img, p, 2, (255, 0, 0), -1)
        for p in right_contour_left:
            cv2.circle(contour_right_img, p, 2, (255, 0, 0), -1)
        
        cv2.imwrite(get_ouput_path("contour_left.png"),contour_left_img)
        cv2.imwrite(get_ouput_path("contour_right.png"),contour_right_img)

        img_left_test_x_left = int(left_poly_model_left.predict([[img_left_test_y]])[0])
        img_right_test_x_left = int(right_poly_model_left.predict([[img_right_test_y]])[0])
        img_left_test_x_right = int(left_poly_model_right.predict([[img_left_test_y]])[0])
        img_right_test_x_right = int(right_poly_model_right.predict([[img_right_test_y]])[0])
        return img_left_test_x_left, img_right_test_x_left,img_left_test_x_right,img_right_test_x_right

    def find_matching(left_center, right_center, img_left, img_right,camTransform:Transform, template_size=30,debug=True):
        image_height, image_width = img_left.shape[:2]

        left_image_x,left_image_y =left_center
        right_image_x,right_image_y = right_center
        
        assert(template_size//2 == int(template_size/2.))
        image_search_size = int(template_size*4)
        
        template = img_left[left_image_y-template_size:left_image_y+template_size,left_image_x-template_size:left_image_x+template_size]
        
        rightImageExtract = img_right[
            right_image_y-image_search_size*2:right_image_y+image_search_size*2,
            right_image_x-image_search_size:right_image_x+image_search_size
            ]
        
        if debug and False:
            cv2.imshow("template",template)
            cv2.imwrite(get_ouput_path( "template.png"), template)
            cv2.imshow("rightImageExtract", rightImageExtract)
            cv2.imwrite(get_ouput_path( "rightImageExtract.png"), rightImageExtract)
        
        template_height, template_height = template.shape[:2]
        searched_zone_height, searched_zone_width = rightImageExtract.shape[:2]
        # All the 6 methods for comparison in a list
        methods = ['TM_CCOEFF', 'TM_CCOEFF_NORMED', 'TM_CCORR',
                'TM_CCORR_NORMED', 'TM_SQDIFF', 'TM_SQDIFF_NORMED']
        
        results=[]
        for meth in methods:
            img = rightImageExtract.copy()
            method = getattr(cv2, meth)
        
            # Apply template Matching
            res = cv2.matchTemplate(img,template,method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
            if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                top_left = min_loc
                score = min_val
            else:
                top_left = max_loc
                score = max_val

            bottom_right = (top_left[0] + template_height, top_left[1] + template_height)

            looked_area_center_x = searched_zone_width//2
            looked_area_center_y = searched_zone_height//2

            found_box_in_area_center_x = top_left[0] + template_height // 2
            found_box_in_area_center_y = top_left[1] + template_height // 2

            right_image_matched_x=right_image_x+found_box_in_area_center_x-looked_area_center_x
            right_image_matched_y=right_image_y+found_box_in_area_center_y-looked_area_center_y
            
            if debug:
                copyLeft = img_left.copy()
                copyRight= img_right.copy()
                cv2.circle(copyLeft, [left_image_x,left_image_y], 20, (255, 0, 0), -1)
                cv2.circle(copyRight, [right_image_matched_x,right_image_matched_y], 20, (255, 0, 0), -1)
                both = np.concatenate((copyLeft, copyRight), axis=0)
                cv2.imwrite(get_ouput_path( f"both{meth}.png"), both)

            P1, P2, residual_distance_in_m = get_3d_point_cam1_2_from_coordinates(
                tuple([left_image_x,left_image_y]),
                tuple([right_image_matched_x, right_image_matched_y]),
                image_width=image_width,
                image_height=image_height,
                t=camTransform.translationVector,
                R=camTransform.rotationMatrix
            )
            if debug and False:
                print("*"*20)
                print(meth, score)
                print("residual_distance_in_m",residual_distance_in_m)
                print(right_image_matched_x, right_image_matched_y)
                print(P1,P2)
                cv2.rectangle(img,top_left, bottom_right, 255, 2)
            
                plt.subplot(121),plt.imshow(res,cmap = 'gray')
                plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
                plt.subplot(122),plt.imshow(img,cmap = 'gray')
                plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
                plt.suptitle(meth)
            
                plt.show()
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            results.append([meth, score, residual_distance_in_m, P1, P2])
            
        return results

    computed_values =[{
        "img_left_test_y" : 1497,
        "img_left_test_x_left" : 2404,
        "img_right_test_x_left" : 2333,
        "img_left_test_x_right" : 2947,
        "img_right_test_x_right": 2789
    }]


    def find_values_by_y(computed_values, target_y):
        """
        Find the dictionary in computed_values where img_left_test_y matches the target_y.

        Parameters:
        - computed_values: List of dictionaries containing the values.
        - target_y: The value of img_left_test_y to search for.

        Returns:
        - A dictionary where img_left_test_y matches target_y, or None if not found.
        """
        for values in computed_values:
            if values["img_left_test_y"] == target_y:
                return values['img_left_test_x_left'],values['img_right_test_x_left'],values['img_left_test_x_right'],values['img_right_test_x_right']
        return None  # Return None if no match is found
    
    optimized_params = load_calibration_params(get_ouput_path("5_2.csv"))
    optimized_t = optimized_params[:3]
    optimized_t*=1.12/optimized_t[0]
    tx,ty, tz = optimized_t
    pitch, yaw, roll = optimized_params[3:]
    cam_right_transform= Transform(tx, ty, tz, pitch, yaw, roll)
    print(cam_right_transform)
    
    img_left_test_y = 1497
    img_right_test_y = img_left_test_y
    result = find_values_by_y(computed_values, img_left_test_y)

    if result is None:
        img_left_test_x_left, img_right_test_x_left,img_left_test_x_right,img_right_test_x_right= compute_xs_from_y(img_left=img_left,img_right=img_right,img_left_test_y=img_left_test_y)
    else:
        img_left_test_x_left, img_right_test_x_left,img_left_test_x_right,img_right_test_x_right = result
    
    print("used img_left_test_y and img_left_test_x,img_right_test_x ")
    print(img_left_test_y,img_left_test_x_left, img_right_test_x_left,img_left_test_x_right,img_right_test_x_right)

    results_left =find_matching(
        [img_left_test_x_left,img_left_test_y],
        [img_right_test_x_left,img_right_test_y],
        img_left,img_right,cam_right_transform,template_size=50)
    
    results_right =find_matching(
        [img_left_test_x_right,img_left_test_y],
        [img_right_test_x_right,img_right_test_y],
        img_left,img_right,cam_right_transform,template_size=50)
    
    for i in range(len(results_left)):
        meth, score, residual_distance_in_m, P1_L, P2_L= results_left[i]
        meth, score, residual_distance_in_m, P1_LR, P2_R= results_right[i]
        print(meth,np.linalg.norm(np.array(P1_L)-np.array(P1_LR)))

    print("done")