from typing import List, Tuple
import numpy as np
import numpy.typing as npt

def compute_3d_position_from_disparity(x: float, y: float, disparity: float, fx: float,fy: float, cx: float, cy: float, baseline: float,z0:float) -> List[float]:    
    """
    Compute the 3D position of a point in the disparity value.

    Parameters:
    - x, y: Pixel coordinates in the image
    - disparity_map: Disparity map (2D array)
    - fx,fy: Focal length of the camera
    - cx, cy: Principal point coordinates (optical center)
    - baseline: Distance between the two camera centers

    Returns:
    - (X, Y, Z): 3D coordinates of the point
    """

    if disparity <= 0:
        raise ValueError("Disparity must be positive and non-zero.")

    Z = (fx * baseline) / (disparity + fx*baseline/z0)
    X = ((x - cx) * Z) / fx
    Y = ((y - cy) * Z) / fy

    return [X, Y, Z]




def compute_3d_position_from_disparity_map(x: float, y: float, disparity_map: npt.NDArray[np.float32], fx: float,fy: float, cx: float, cy: float, baseline: float,z0:float) -> Tuple[List[float], float]:
    """
    Compute the 3D position of a point in the disparity map.

    Parameters:
    - x, y: Pixel coordinates in the image
    - disparity_map: Disparity map (2D array)
    - fx,fy: Focal length of the camera
    - cx, cy: Principal point coordinates (optical center)
    - baseline: Distance between the two camera centers

    Returns:
    - (X, Y, Z): 3D coordinates of the point
    - disparity: Disparity value at the point
    """
    u = int(x)
    v = int(y)
    
    disparity = disparity_map[v, u]
    [X, Y, Z]= compute_3d_position_from_disparity(x,y,disparity,fx,fy,cx,cy,baseline)

    return [X, Y, Z], disparity

