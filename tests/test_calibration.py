import os
import unittest
import cv2
import numpy as np
import random
from bootstrap import get_image_folder_path, set_paths
set_paths()
from src.calibration.eac import calibrate_left_right


class TestCalibrateLeftRight(unittest.TestCase):

    def setUp(self):
        # Set a fixed seed for reproducibility
        random.seed(42)
        np.random.seed(42)

        # Load or create test images
        image_folder= get_image_folder_path()
        self.imLeft = cv2.imread(os.path.join(image_folder, 'D_P1_CAM_G_0_EAC.png' ))
        self.imRight = cv2.imread(os.path.join(image_folder, 'D_P1_CAM_D_0_EAC.png' ))

        # Ensure images are loaded correctly
        self.assertIsNotNone(self.imLeft, "Left image not loaded")
        self.assertIsNotNone(self.imRight, "Right image not loaded")

        # Define the initial parameters and bounds
        angle_max = np.pi*10./180.
        dt_max = 0.12001
        self.bnds = ((-angle_max, angle_max), (-angle_max, angle_max),(-angle_max, angle_max),(1.11, 1.13),(-dt_max,dt_max),(-dt_max,dt_max))
        self.initial_params = [0, 0, 0, 1.12,0,0]
        self.inlier_threshold = 0.01

    def test_calibrate_left_right(self):
        # Call the function
        result = calibrate_left_right(self.imLeft, self.imRight, self.initial_params, self.bnds, self.inlier_threshold, verbose=False)
        print(result)
        # Assertions to verify the output
        self.assertIn("max_inliers", result)
        self.assertIn("R", result)
        self.assertIn("t", result)
        self.assertIn("params", result)
        self.assertEqual(len(result["params"]), 6)
        self.assertTrue(isinstance(result["R"], np.ndarray))
        self.assertTrue(isinstance(result["t"], np.ndarray))

        # Ensure that some matches were found and the calibration process completed
        self.assertGreater(result["max_inliers"], 0, "No inliers found, calibration might have failed")

        # Further evaluate the found R and t
        R = result["R"]
        t = result["t"]

        print(R)
        print(t)

        # Compare the found solution with the ground truth
        # This part depends on the ground truth values for R and t
        # For example, if you have R_gt and t_gt from a calibration process:
        # Assuming you have ground truth R_gt and t_gt, you can compare:
        R_gt = [[ 0.99686161, -0.01548253,  0.07763524],
       [ 0.01775139,  0.99943273, -0.02862019],
       [-0.07714808,  0.0299085 ,  0.99657095]]
        t_gt = [ 1.11981549,  0.01911424, -0.03470565]

        # Calculate the difference between the found and the ground truth
        R_diff = np.linalg.norm(R_gt - R)
        t_diff = np.linalg.norm(np.array(t_gt) - np.array(t))

        # Tolerances for the differences
        R_tolerance = 1e-2
        t_tolerance = 1e-2

        self.assertLessEqual(R_diff, R_tolerance, "Rotation matrix R is not within the acceptable range")
        self.assertLessEqual(t_diff, t_tolerance, "Translation vector t is not within the acceptable range")

if __name__ == '__main__':
    unittest.main()
