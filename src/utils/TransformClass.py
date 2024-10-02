from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from scipy.spatial.transform import Rotation as R


@dataclass
class Transform:
    xc:float
    yc:float
    zc:float
    pitch:float
    yaw:float
    roll:float

    def __init__(self, xc, yc, zc,pitch, yaw, roll):
        self.xc = xc
        self.yc = yc
        self.zc = zc
        self.pitch = pitch
        self.yaw = yaw
        self.roll = roll

    def as_array(self)->np.array:
       return np.array([
        self.xc,
        self.yc,
        self.zc,
        self.pitch,
        self.yaw,
        self.roll
    ])

    def as_params(self)->np.array:
       return np.array([
        self.yc,
        self.zc,
        self.pitch,
        self.yaw,
        self.roll
    ])

    def scale_translation_from_x(self, baseline:float)->None:
        ratio = baseline/self.xc
        self.xc *= ratio
        self.yc *= ratio
        self.zc *= ratio
        
    @property
    def rotationMatrix(self)->np.array:
        rotation = R.from_euler('xyz', [self.pitch,self.yaw,self.roll ], degrees=False)
        rotation_matrix = rotation.as_matrix()
        return rotation_matrix
    
    @property
    def translationVector(self)->np.array:
        return np.array([self.xc, self.yc, self.zc])  
    

@dataclass
class TransformBounds:
    baseline:float
    dt_max_y:float
    dt_max_z: float
    angle_max: float

    def __init__(self, baseline, dt_max_y,dt_max_z, angle_max):
        self.baseline = baseline
        self.dt_max_y = dt_max_y
        self.dt_max_z = dt_max_z
        self.angle_max = angle_max  

    def get_bounds(self) -> Tuple[Tuple[float, float], ...]:
        """
        Constructs the bounds for the optimization parameters in the format of tuples for each parameter.

        Returns:
            Tuple[Tuple[float, float], ...]: A tuple of tuples where each tuple represents the (lower_bound, upper_bound)
            for a particular parameter.
        """
        # Translation bounds
        ty_bounds = (-self.dt_max_y, self.dt_max_y)
        tz_bounds = (-self.dt_max_z, self.dt_max_z)

        # Rotation bounds (in radians)
        roll_bounds = (-self.angle_max, self.angle_max)
        pitch_bounds = (-self.angle_max, self.angle_max)
        yaw_bounds = (-self.angle_max, self.angle_max)
        bounds = (
            ty_bounds,
            tz_bounds,
            roll_bounds,
            pitch_bounds,
            yaw_bounds
        )

        return bounds