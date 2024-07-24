import numpy as np
import cv2

from src.auto_calibration_utils import compute_auto_calibration_for_images


def rectify_images(img1,img2):
    #1084 1051 => 648 525
    focal_length,_,refined_rvec,refined_tvec =compute_auto_calibration_for_images([img1,img2])
    print(focal_length)
    K=np.array([[focal_length ,  0.,664. ],
    [0., focal_length, 656. ],
    [0.,0.,1. ]])
    K=np.array([[571.92127691, 0.,649.87876627],
 [  0.,561.6543542 , 594.76301074],
 [  0.,0.,1.]])
    A1 = K # Left camera matrix intrinsic
    A2 = K # Right camera matrix intrinsic
    
    RT1 = np.eye(3, 4)

    rvec = np.array([[0.02824257], [0.04876492], [0.01698908]])
    tvec = np.array([[-1.12004838], [0.00416585], [0.00767302]])

    rvec=refined_rvec
    tvec=refined_tvec

    # Convert the rotation vector to a rotation matrix
    R, _ = cv2.Rodrigues(rvec)
    
    # Create the 3x4 extrinsic matrix by combining R and tvec
    RT2 = np.hstack((R, tvec))
    
    # Original projection matrices
    Po1 = A1.dot( RT1 )
    Po2 = A2.dot( RT2 )

    # Camera centers (world coord.)
    C1 = -np.linalg.inv(Po1[:,:3]).dot(Po1[:,3])
    C2 = -np.linalg.inv(Po2[:,:3]).dot(Po2[:,3])

    # Transformations
    T1to2 = C2 - C1 # Translation from first to second camera
    R1to2 = RT2[:,:3].dot(np.linalg.inv(RT1[:,:3])) # Rotation from first to second camera (3x3)

    R1, R2, Pn1, Pn2, _, _, _ = cv2.stereoRectify(A1, np.zeros((1,5)), A2, np.zeros((1,5)), (img1.shape[1], img1.shape[0]), R1to2, T1to2, alpha=-1 )

    # Rectify1 = R1.dot(np.linalg.inv(A1))
    # Rectify2 = R2.dot(np.linalg.inv(A2))

    mapL1, mapL2 = cv2.initUndistortRectifyMap(A1, np.zeros((1,5)), R1, Pn1, (img1.shape[1], img1.shape[0]), cv2.CV_32FC1)
    mapR1, mapR2 = cv2.initUndistortRectifyMap(A2, np.zeros((1,5)), R2, Pn2, (img2.shape[1], img2.shape[0]), cv2.CV_32FC1)

    img1_rect = cv2.remap(img1, mapL1, mapL2, cv2.INTER_LINEAR)
    img2_rect = cv2.remap(img2, mapR1, mapR2, cv2.INTER_LINEAR)
    return img1_rect, img2_rect


