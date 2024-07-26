import numpy as np


def create_Q_matrix(focal_length:float, baseline:float, c_x:float, c_y:float, c_x_prime=None):
    """
    Create the Q matrix for stereo vision.
    
    Parameters:
    focal_length (float): The focal length of the cameras.
    baseline (float): The distance between the two cameras.
    c_x (float): The x-coordinate of the principal point in the left camera.
    c_y (float): The y-coordinate of the principal point in the left camera.
    c_x_prime (float, optional): The x-coordinate of the principal point in the right camera. 
                                 If None, it's assumed to be the same as c_x.

    Returns:
    np.ndarray: The 4x4 Q matrix.
    """
    if c_x_prime is None:
        c_x_prime = c_x
    
    Q = np.array([[1, 0, 0, -c_x],
                  [0, 1, 0, -c_y],
                  [0, 0, 0, focal_length],
                  [0, 0, -1/baseline, (c_x - c_x_prime)/baseline]])
    
    return Q