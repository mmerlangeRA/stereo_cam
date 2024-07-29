import random
import cv2
from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import RANSACRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from src.pidnet.main import segment_image
from src.utils.geo import create_Q_matrix
from src.depth_estimation.depth_estimator import InputPair
from src.depth_estimation.selective_igev import Selective_igev
from src.utils.path_utils import get_static_folder_path
#from sklearn.cluster import DBSCAN

def compute_3d_position(x:float, y:float, disparity_map, f, cx, cy, baseline):
    """
    Compute the 3D position of a point in the disparity map.

    Parameters:
    - u, v: Pixel coordinates in the image
    - disparity: Disparity value at pixel (u, v)
    - f: Focal length of the camera
    - cx, cy: Principal point coordinates (optical center)
    - baseline: Distance between the two camera centers

    Returns:
    - (X, Y, Z): 3D coordinates of the point
    """
    u=int(x)
    v=int(y)
    disparity = disparity_map[v, u]
    if disparity <= 0:
        raise ValueError("Disparity must be positive and non-zero.")

    Z = (f * baseline) / disparity
    X = ((u - cx) * Z) / f
    Y = ((v - cy) * Z) / f

    return [X, Y, Z],disparity

def fit_polynomial_ransac(x, y, degree=2,residual_threshold=10):
    poly_model = make_pipeline(PolynomialFeatures(degree), RANSACRegressor(residual_threshold=residual_threshold))
    poly_model.fit(x[:, np.newaxis], y)
    inlier_mask = poly_model.named_steps['ransacregressor'].inlier_mask_
    return poly_model, inlier_mask

def find_polynomial_curves(contour, img):
    contour_points = contour[:, 0, :]
    x = contour_points[:, 0]
    y = contour_points[:, 1]
    # Fit the first polynomial curve using RANSAC
    first_poly_model, inliers_first = fit_polynomial_ransac(y, x)

    # Remove inliers to find the second polynomial curve
    y_inliers_first= y[inliers_first]
    x_outliers = x[~inliers_first]
    y_outliers = y[~inliers_first]
    # Fit the second polynomial curve using RANSAC
    second_poly_model, inliers_second = fit_polynomial_ransac(y_outliers,x_outliers)
    y_inliers_second= y_outliers[inliers_second]
    # Generate x values for plotting the polynomial curves
    y_range = np.linspace(np.min(y), np.max(y), 500)

    if img is not None:
        # Predict y values using the polynomial models
        x_first_poly = first_poly_model.predict(y_range[:, np.newaxis])
        x_second_poly = second_poly_model.predict(y_range[:, np.newaxis])

        first_coefficients = first_poly_model.named_steps['ransacregressor'].estimator_.coef_
        second_coefficients = second_poly_model.named_steps['ransacregressor'].estimator_.coef_
        nb_inliers_first = np.sum(inliers_first)
        nb_inliers_second = np.sum(inliers_second)
        print(f"First polynomial coefficients: {first_coefficients}, Inliers: {nb_inliers_first}")
        print(f"First polynomial coefficients: {second_coefficients}, Inliers: {nb_inliers_second}")
        # Plot the polynomial curves on the image
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.plot(x_first_poly,y_range, color='red', linewidth=2, label='First Polynomial')
        plt.plot(x_second_poly,y_range, color='blue', linewidth=2, label='Second Polynomial')
        plt.scatter(x,y, color='yellow', s=5, label='Contour Points')
        plt.legend()
        plt.title('Polynomial Curves Fit to Contour Points')
        plt.show()

    return first_poly_model, second_poly_model, y_inliers_first, y_inliers_second
    

def get_road_edges(imgL:cv2.typing.MatLike, imgR:cv2.typing.MatLike):

    test_igev = Selective_igev(None,None)
    input = InputPair(left_image=imgL,right_image=imgR,status="started", calibration=None)
    stereo_output = test_igev.compute_disparity(input)
    disparity_map = stereo_output.disparity_pixels
    cv2.imwrite(get_static_folder_path("disparity.png"), disparity_map)

    focal_length = 700.0  # in pixels
    baseline = 1.12  # in meters
    c_x = 331.99987265  # principal point x-coordinate
    c_y = 387.5000997 # principal point y-coordinate

    depth_map = (focal_length * baseline) / (disparity_map + 1e-6)
    # Assuming you have a function to perform semantic segmentation
    segmented_image,pred = segment_image(imgL)
    # Create a mask for the road class
    road_mask = (pred == 0).astype(np.uint8)
    # Check if the mask has the same dimensions as the segmented image
    assert road_mask.shape == segmented_image.shape[:2], "Mask size does not match the image size."
 
    masked_segmented = cv2.bitwise_and(segmented_image, segmented_image, mask=road_mask)

    cv2.imshow('img', masked_segmented)
    cv2.imwrite(get_static_folder_path("maksed_road.png"), masked_segmented)

 
    cv2.imwrite(get_static_folder_path("disparity_map.png"), disparity_map)
    cv2.imshow('depth', depth_map)

    cv2.imwrite(get_static_folder_path("depth_map.png"), depth_map)

    gray = cv2.cvtColor(masked_segmented, cv2.COLOR_BGR2GRAY)
    # edged = cv2.Canny(gray, 10, 100)
    # cv2.imshow('edged', edged)
    # cv2.imwrite(get_static_folder_path("edges.png"), edged)

    contour_image = np.zeros_like(masked_segmented)
    ret, thresh = cv2.threshold(gray, 1, 255, 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    # apply the dilation operation to the edged image
    thresh=cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    cv2.imshow('thresh', thresh)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(f'found {len(contours)}')
    # Draw contours with random colors
    for contour in contours:
        # Generate a random color
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.drawContours(contour_image, [contour], -1, color, 3)
    
    contour = max(contours, key=cv2.contourArea)
    first_poly_model, second_poly_model, y_inliers_first, y_inliers_second = find_polynomial_curves(contour, imgL)
    min_y_inliers_first=np.min(y_inliers_first)
    min_y_inliers_second=np.min(y_inliers_first)
    max_y_inliers_first=np.max(y_inliers_first)
    max_y_inliers_second=np.max(y_inliers_second)

    minY = max(min_y_inliers_first, min_y_inliers_second)
    maxY = min(max_y_inliers_first, max_y_inliers_second)

    if maxY<minY:
        print("no road")

    distances=[]
    points=[]
    for y in range(minY+60, maxY-60):
        x_first_poly = first_poly_model.predict([[y]])[0]
        x_second_poly = second_poly_model.predict([[y]])[0]
        p1,d1 = compute_3d_position(x_first_poly,y, disparity_map, focal_length, c_x, c_y, baseline)
        p2,d2 = compute_3d_position(x_second_poly,y, disparity_map, focal_length, c_x, c_y, baseline)
        if np.abs(d2-d1)>10:
            print(d2,d1)
        points.append([p1,p2])
        distances.append(np.linalg.norm(np.array(p1) - np.array(p2)))
    
    print(np.mean(distances))
    print(points)
    print(distances)
    
    cv2.imshow('contours', contour_image)
    #cv2.waitKey(0)
    cv2.imwrite(get_static_folder_path("contours.png"), contour_image)





