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

    _rotationMatrix:np.array = None
    _inverseRotationMatrix:np.array = None

    def __init__(self, xc:float=0., yc:float=0., zc:float=0.,pitch:float=0., yaw:float=0., roll:float=0.):
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
        if self._rotationMatrix is None:
            self._rotationMatrix =  R.from_euler('xyz', [self.pitch,self.yaw,self.roll ], degrees=False)
            self._rotationMatrix = self._rotationMatrix.as_matrix()
        return self._rotationMatrix
    
    @property
    def inverseRotationMatrix(self)->np.array:
        if self._inverseRotationMatrix is None:
            self._inverseRotationMatrix = self.rotationMatrix.T
        return self._inverseRotationMatrix
    
    @property
    def translationVector(self)->np.array:
        return np.array([self.xc, self.yc, self.zc]) 

    @property
    def rotationVector(self)->np.array:
        return np.array([self.pitch, self.yaw, self.roll ]) 
    

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