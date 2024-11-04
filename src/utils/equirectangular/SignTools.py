import numpy as np


class SignMatcherTool:
    def normalize_to_z1(self,x:float, y:float, z:float,to_value=1.):
        """
        Normalizes Cartesian coordinates so that z = 1.
        """
        scale = to_value / z
        x_new = x * scale
        y_new = y * scale
        z_new = to_value
        return x_new, y_new, z_new

    def scale_positions(self,x:float, y:float, z:float, ratio:float):
        """
        Scales Cartesian coordinates by a given ratio.
        """
        x_scaled = x * ratio
        y_scaled = y * ratio
        z_scaled = z * ratio
        return x_scaled, y_scaled, z_scaled
