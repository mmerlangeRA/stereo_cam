import cv2
from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import RANSACRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from typing import Tuple, List, Optional
import numpy.typing as npt

def fit_polynomial_ransac(x: npt.NDArray[np.float32], y: npt.NDArray[np.float32], degree: int = 2, residual_threshold: float = 10) -> Tuple[make_pipeline, npt.NDArray[np.bool_]]:
    """
    Fits a polynomial curve to the given x, y data using RANSAC.

    Parameters:
    - x: Array of x values.
    - y: Array of y values.
    - degree: Degree of the polynomial.
    - residual_threshold: RANSAC residual threshold.

    Returns:
    - poly_model: Fitted polynomial model.
    - inlier_mask: Mask of inliers detected by RANSAC.
    """
    if len(x) != len(y) or len(x)==0:
        raise ValueError("x and y must have the same non zero length")
    poly_model = make_pipeline(PolynomialFeatures(degree), RANSACRegressor(residual_threshold=residual_threshold))
    poly_model.fit(x[:, np.newaxis], y)
    inlier_mask = poly_model.named_steps['ransacregressor'].inlier_mask_
    return poly_model, inlier_mask

def find_best_2_polynomial_curves(contour: npt.NDArray[np.int32],degree=2,max_pixel_distance=10) -> Tuple[make_pipeline, make_pipeline, npt.NDArray[np.int32], npt.NDArray[np.int32]]:
    """
    Finds and plots two polynomial curves fit to the given contour points.

    Parameters:
    - contour: Array of contour points.
    - img: Image to plot the curves on. Optional.

    Returns:
    - first_poly_model: Fitted model for the first polynomial curve.
    - second_poly_model: Fitted model for the second polynomial curve.
    - y_inliers_first: Inlier y-values for the first polynomial.
    - y_inliers_second: Inlier y-values for the second polynomial.
    """
    contour_points = contour[:, 0, :]
    x = contour_points[:, 0]
    y = contour_points[:, 1]
    # Fit the first polynomial curve using RANSAC
    first_poly_model, inliers_first = fit_polynomial_ransac(y, x,degree,residual_threshold=max_pixel_distance)

    # Remove inliers to find the second polynomial curve
    y_inliers_first = y[inliers_first]
    x_outliers = x[~inliers_first]
    y_outliers = y[~inliers_first]
    # Fit the second polynomial curve using RANSAC
    second_poly_model, inliers_second = fit_polynomial_ransac(y_outliers, x_outliers,degree,residual_threshold=max_pixel_distance)
    y_inliers_second = y_outliers[inliers_second]


    return first_poly_model, second_poly_model, y_inliers_first, y_inliers_second

