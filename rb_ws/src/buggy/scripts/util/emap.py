import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator

class EMap:
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)

        self.X = df['X'].values
        self.Y = df['Y'].values
        self.Z = df['VALUE'].values

        # LinearNDInterpolator naturally handles unstructured data and missing points.
        # It takes a moment to build the triangulation, but queries are extremely fast.
        points = np.column_stack((self.X, self.Y))
        self._interpolator = LinearNDInterpolator(points, self.Z)

    def elevation(self, x, y):
        """
        Returns the continuous elevation at (x, y). 
        Supports both scalar inputs and vectorized numpy arrays.
        """
        return self._interpolator(x, y)

    def grad(self, x, y, h=0.1):
        """
        Returns the gradient (dz/dx, dz/dy) using central finite difference.
        'h' is the step size in meters.
        """
        # Central difference: (f(x+h) - f(x-h)) / 2h
        dzdx = (self.elevation(x + h, y) - self.elevation(x - h, y)) / (2 * h)
        dzdy = (self.elevation(x, y + h) - self.elevation(x, y - h)) / (2 * h)
        
        return dzdx, dzdy