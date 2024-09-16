from abc import abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple
import time
import json
import numpy as np

@dataclass
class Calibration:
    width: int
    height: int
    fx: float
    fy: float
    cx0: float # cx is the only one that can differ between both cameras.
    cx1: float
    cy: float
    baseline_meters: float
    z0: float = float('-inf')
    depth_range: Tuple[float] = (0.3, 20.0)
    left_image_rect_normalized: np.ndarray = field(default_factory=lambda: np.array([0., 0., 1., 1.]))  # origin, size in percent of image size
    comment: str = ""

    def to_json(self) -> str:
        # Convert the dataclass to a dictionary
        dict_representation = asdict(self)

        # Convert any numpy arrays to lists
        for key, value in dict_representation.items():
            if isinstance(value, np.ndarray):
                dict_representation[key] = value.tolist()

        return json.dumps(dict_representation)

    def from_json(json_str):
        d = json.loads(json_str)
        return Calibration(**d)

    def downsample(self, new_width: int, new_height: int):
        sx = new_width / self.width
        sy = new_height / self.height
        self.width = new_width
        self.height = new_height
        self.fx *= sx
        self.fy *= sy
        self.cx0 *= sx
        self.cx1 *= sx
        self.cy *= sy

@dataclass
class InputPair:
    left_image: np.ndarray
    right_image: np.ndarray
    calibration: Calibration
    status: str
    input_disparity: np.ndarray = None

    def has_data(self):
        return self.left_image is not None

@dataclass
class StereoOutput:
    disparity_pixels: np.ndarray
    color_image_bgr: np.ndarray
    computation_time: float
    point_cloud: Any = None
    disparity_color: np.ndarray = None

@dataclass
class IntParameter:
    description: str
    value: int
    min: int
    max: int
    to_valid: Any = lambda x: x # default is to just accept anything

    def set_value (self, x: int):
        self.value = self.to_valid(x)

@dataclass
class EnumParameter:
    description: str
    index: int # index in the list
    values: List[str]

    def set_index (self, idx: int):
        self.index = idx

    def set_value (self, value):
        self.index = self.values.index(value)

    @property
    def value(self) -> str:
        return self.values[self.index]

@dataclass
class Config:
    models_path: Path

class StereoMethod:
    def __init__(self, name: str,  parameters: Dict, config: Config):
        self.name = name
        self.parameters = parameters
        self.config = config

    def reset_defaults(self):
        pass

    @abstractmethod    
    def compute_disparity(self, input: InputPair) -> StereoOutput:
        """Return the disparity map in pixels and the actual computation time.
        
        Both input images are assumed to be rectified.
        """
        return StereoOutput(None, None, None, None)

    def depth_meters_from_disparity(self,disparity_pixels: np.ndarray, calibration: Calibration):
        old_seterr = np.seterr(divide='ignore')
        dcx = np.float32(calibration.cx0 - calibration.cx1)
        print(dcx)
        print(calibration.baseline_meters)
        depth_meters = np.float32(calibration.baseline_meters * calibration.fx) / (disparity_pixels - dcx)
        print(np.min(depth_meters))
        print(np.max(depth_meters))
        depth_meters = np.nan_to_num(depth_meters)
        depth_meters[disparity_pixels < 0.] = -1.0
        np.seterr(**old_seterr)
        print(np.min(depth_meters))
        print(np.max(depth_meters))
        return depth_meters

    def disparity_from_depth_meters(self,depth_meters: np.ndarray, calibration: Calibration):
        old_seterr = np.seterr(divide='ignore')
        dcx = np.float32(calibration.cx0 - calibration.cx1)
        disparity_pixels = (np.float32(calibration.baseline_meters * calibration.fx) / depth_meters) + dcx
        disparity_pixels = np.nan_to_num(disparity_pixels)
        np.seterr(**old_seterr)
        return disparity_pixels
