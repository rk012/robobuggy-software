import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator, RectBivariateSpline
from matplotlib.path import Path

class EMap:
    def __init__(self, csv_path, inner_path=None, outer_path=None, resolution=0.5, tolerance=2.0):
        print(f"EMap: Loading CSV from {csv_path}...")
        df = pd.read_csv(csv_path)

        if outer_path is not None and inner_path is not None:
            print("EMap: Stitching boundaries into a continuous track ribbon...")
            
            # Combine the paths into a single continuous boundary loop
            # (Outer going forward, Inner going backward, closing back to start)
            polygon_vertices = np.vstack([
                outer_path, 
                inner_path[::-1],
                outer_path[0]
            ])
            track_poly = Path(polygon_vertices)

            # 1. Fast bounding box filter based on the total vertices
            min_x = np.min(polygon_vertices[:, 0]) - tolerance
            max_x = np.max(polygon_vertices[:, 0]) + tolerance
            min_y = np.min(polygon_vertices[:, 1]) - tolerance
            max_y = np.max(polygon_vertices[:, 1]) + tolerance

            bbox_mask = (df['X'] >= min_x) & (df['X'] <= max_x) & \
                        (df['Y'] >= min_y) & (df['Y'] <= max_y)
            
            df_bbox = df[bbox_mask]
            
            # 2. Winding-Agnostic Ribbon Filter
            raw_pts = np.column_stack((df_bbox['X'].values, df_bbox['Y'].values))
            
            # Using the boolean OR (|) trick ensures that the tolerance expands 
            # the polygon outward, bypassing any clockwise/counter-clockwise normal vector flips
            poly_mask = track_poly.contains_points(raw_pts, radius=tolerance) | \
                        track_poly.contains_points(raw_pts, radius=-tolerance)
            
            original_len = len(df)
            df = df_bbox[poly_mask]
            print(f"EMap: Kept {len(df)} / {original_len} raw points (+ {tolerance}m tolerance).")

        self.X = df['X'].values
        self.Y = df['Y'].values
        self.Z = df['VALUE'].values

        # 1. Build the interpolator on the highly filtered point set
        print("EMap: Building initial unstructured mesh...")
        lin_interp = LinearNDInterpolator(np.column_stack((self.X, self.Y)), self.Z)

        # 2. Define the dense grid bounding box
        min_x_pts, max_x_pts = np.min(self.X), np.max(self.X)
        min_y_pts, max_y_pts = np.min(self.Y), np.max(self.Y)
        
        self.x_grid = np.arange(min_x_pts, max_x_pts, resolution)
        self.y_grid = np.arange(min_y_pts, max_y_pts, resolution)
        X_mesh, Y_mesh = np.meshgrid(self.x_grid, self.y_grid, indexing='ij')

        print("EMap: Masking empty areas of the grid to save baking time...")
        flat_grid = np.column_stack((X_mesh.flatten(), Y_mesh.flatten()))
        
        if outer_path is not None and inner_path is not None:
            # Apply the exact same winding-agnostic mask to the dense grid
            grid_mask = track_poly.contains_points(flat_grid, radius=tolerance) | \
                        track_poly.contains_points(flat_grid, radius=-tolerance)
        else:
            grid_mask = np.ones(len(flat_grid), dtype=bool)

        # 3. Bake the interpolator ONLY onto the track points
        valid_pts = np.sum(grid_mask)
        print(f"EMap: Baking {valid_pts} track points into spline grid...")
        
        Z_flat = np.full(len(flat_grid), np.nan)
        Z_flat[grid_mask] = lin_interp(flat_grid[grid_mask])
        Z_grid = Z_flat.reshape(X_mesh.shape)

        # Flatten NaN holes to the mean track elevation
        mean_z = np.nanmean(Z_grid)
        Z_grid = np.nan_to_num(Z_grid, nan=mean_z)

        # 4. Fit the Spline
        print("EMap: Fitting high-speed spline...")
        self._spline = RectBivariateSpline(self.x_grid, self.y_grid, Z_grid, kx=3, ky=3)
        print("EMap: Ready.")

    def elevation(self, x, y):
        """Returns the continuous elevation at (x, y)."""
        res = self._spline(x, y, grid=False)
        return float(res) if np.isscalar(x) else res

    def grad(self, x, y):
        """Returns the exact analytical gradient (dz/dx, dz/dy) from the Spline."""
        dzdx = self._spline(x, y, dx=1, dy=0, grid=False)
        dzdy = self._spline(x, y, dx=0, dy=1, grid=False)
        
        if np.isscalar(x):
            return float(dzdx), float(dzdy)
        return dzdx, dzdy