import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import utm
from util.constants import Constants
from util.emap import EMap

def load_json_to_utm(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    utm_points = []
    for pt in data:
        lat = pt['lat']
        lon = pt['lon']
        easting, northing, _, _ = utm.from_latlon(lat, lon, force_zone_number=Constants.UTM_ZONE_NUM)
        utm_points.append([easting, northing])
        
    return np.array(utm_points)

def main():
    parser = argparse.ArgumentParser(description="Visualize buggy course elevation and gradients.")
    parser.add_argument("course_name", help="Base name of the course (e.g., 'paths/cut_square')")
    parser.add_argument("csv_path", help="Path to the Elevation CSV file")
    args = parser.parse_args()

    outer_file = f"{args.course_name}_outer.json"
    inner_file = f"{args.course_name}_inner.json"

    print(f"Loading constraints from {outer_file} and {inner_file}...")
    outer_points = load_json_to_utm(outer_file)
    inner_points = load_json_to_utm(inner_file)

    polygon_vertices = np.vstack([
        outer_points, 
        inner_points[::-1],
        outer_points[0]
    ])
    track_path = Path(polygon_vertices)

    print("Initializing EMap (building triangulation)...")
    emap = EMap(args.csv_path, inner_points, outer_points)

    print("Generating dense elevation grid...")
    min_x, max_x = np.min(polygon_vertices[:, 0]), np.max(polygon_vertices[:, 0])
    min_y, max_y = np.min(polygon_vertices[:, 1]), np.max(polygon_vertices[:, 1])

    # 1. DENSE GRID (for the underlying color map)
    elev_res = 0.5
    elev_x, elev_y = np.meshgrid(
        np.arange(min_x, max_x, elev_res),
        np.arange(min_y, max_y, elev_res)
    )

    flat_ex = elev_x.flatten()
    flat_ey = elev_y.flatten()
    
    # Filter dense grid to track boundaries
    elev_mask = track_path.contains_points(np.column_stack((flat_ex, flat_ey)))
    valid_ex = flat_ex[elev_mask]
    valid_ey = flat_ey[elev_mask]
    
    print("Querying EMap for elevations...")
    valid_ez = emap.elevation(valid_ex, valid_ey)

    # 2. COARSE GRID (for the gradient arrows to avoid clutter)
    print("Generating coarse gradient grid...")
    grad_res = 4.0  # Calculate a gradient arrow every 4 meters
    grad_x, grad_y = np.meshgrid(
        np.arange(min_x, max_x, grad_res),
        np.arange(min_y, max_y, grad_res)
    )

    flat_gx = grad_x.flatten()
    flat_gy = grad_y.flatten()
    
    # 2. DENSE GRADIENT GRID (for the flow-line look)
    print("Generating gradient grid...")
    # Decreased from 4.0 to 1.5 to pack the arrows much closer together
    grad_res = 1.5  
    grad_x, grad_y = np.meshgrid(
        np.arange(min_x, max_x, grad_res),
        np.arange(min_y, max_y, grad_res)
    )

    flat_gx = grad_x.flatten()
    flat_gy = grad_y.flatten()
    
    # Filter grid to track boundaries
    grad_mask = track_path.contains_points(np.column_stack((flat_gx, flat_gy)))
    valid_gx = flat_gx[grad_mask]
    valid_gy = flat_gy[grad_mask]

    print("Querying EMap for gradients...")
    dzdx, dzdy = emap.grad(valid_gx, valid_gy)

    print("Rendering 2D Map with Vector Field...")
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot the bounding polygon lines for reference
    ax.plot(outer_points[:, 0], outer_points[:, 1], color='black', linewidth=1, label='Outer Bound')
    ax.plot(inner_points[:, 0], inner_points[:, 1], color='black', linewidth=1, label='Inner Bound')

    # Plot the elevation heatmap
    sc = ax.scatter(valid_ex, valid_ey, c=valid_ez, cmap='viridis', s=5, marker='s', edgecolors='none', alpha=1.0)

    # Plot the gradient vector field (Quiver)
    # TWEAKED: Extremely thin shafts, tiny arrowheads, centered on the coordinates
    # ax.quiver(valid_gx, valid_gy, dzdx, dzdy, 
    #           color='red', 
    #           alpha=0.6,           # Increased transparency so the terrain colors still pop
    #           width=0.0015,        # Very thin line shaft
    #           headwidth=2.5,       # Narrow arrowhead
    #           headlength=3.0,      # Short arrowhead length
    #           headaxislength=2.5,  # Matches headlength for a sharp, tiny triangle
    #           label='Gradient (Uphill)')

    # Formatting
    ax.set_title(f'{args.course_name.capitalize()} Course Elevation')
    ax.set_xlabel('Easting (X)')
    ax.set_ylabel('Northing (Y)')
    
    ax.axis('equal')
    fig.colorbar(sc, label='Elevation (m)')
    plt.legend()
    
    plt.show()

if __name__ == "__main__":
    main()