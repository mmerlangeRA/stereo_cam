import cv2
import numpy as np

from python_server.components.pidnet_segementation.main import segment_image
from src.utils.geo import create_Q_matrix

imgL_path = 'left_image.png'
imgR_path = 'left_image.png'
imgL = cv2.imread('left_image.png', 0)
imgR = cv2.imread('right_image.png', 0)


def get_road_edges(imgL, imgR):

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    imgL = clahe.apply(imgL)
    imgR = clahe.apply(imgR)

    # Assuming you have a function to load and run a pre-trained deep learning model
    # This is a placeholder function call
    disparity = run_deep_learning_stereo_model(imgL, imgR)

    focal_length = 718.856  # example value
    baseline = 0.54  # example value in meters
    depth_map = (focal_length * baseline) / (disparity + 1e-6)


    # Assuming you have a function to perform semantic segmentation
    road_mask = segment_image(imgL)

    focal_length = 800.0  # in pixels
    baseline = 0.1  # in meters
    c_x = 320.0  # principal point x-coordinate
    c_y = 240.0  # principal point y-coordinate

    # Create the Q matrix
    Q = create_Q_matrix(focal_length, baseline, c_x, c_y)

    points_3D = cv2.reprojectImageTo3D(disparity, Q)
    road_points_3D = points_3D[road_mask]

    from sklearn.cluster import DBSCAN

    # Cluster the road points to find edges
    clustering = DBSCAN(eps=0.5, min_samples=10).fit(road_points_3D.reshape(-1, 3))
    labels = clustering.labels_

    # Extract clusters corresponding to the road edges
    unique_labels = set(labels)
    road_edges = [road_points_3D[labels == label] for label in unique_labels if label != -1]

    # Find the width by measuring the distance between the two edge clusters
    edge1, edge2 = road_edges[0], road_edges[1]  # assuming two main edges
    width = np.mean(np.linalg.norm(edge1[:, :2] - edge2[:, :2], axis=1))
