import json
import utm
from pathlib import Path
import uuid

from casadi import *
from shapely.geometry import Point, LineString
from scipy.interpolate import interp1d
from numpy import pi

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

outer_curb = load_json_to_utm("paths/buggycourse_outer.json")
inner_curb = load_json_to_utm("paths/buggycourse_inner.json")

# start position
# everything is shifted to relative offsets from this, better floating point precision during optim
x0, y0 = 589686.3567632461, 4477152.974378172

N = 250

# physical constants
g = 9.807

wheelbase = Constants.WHEELBASE_SC
m = 58.967  # kg
I = 21.847274476  # kg, m, ellipsoid + ballpark measurement

# Friction coefficients
c_rr = 0.02  # rolling resistance, const, 
c_v = 0.15   # linear drag (proportional to v)
c_a = 0.025  # viscous drag (proportional to v^2) 
c_s = 0.77   # steering drag (proportional to v^2 * sin^2(d) )

# load/localize curbs, create signed distance field grid

# shift these to local coordinate frame
local_outer = outer_curb - [x0, y0]
local_inner = inner_curb - [x0, y0]

# Find the bounding box of the entire track
all_pts = np.vstack((local_outer, local_inner))
min_x, min_y = np.min(all_pts, axis=0) - 10.0 # Add 10m margin
max_x, max_y = np.max(all_pts, axis=0) + 10.0

# Create strictly increasing 1D arrays for CasADi
resolution = 1.0 # 1 meter grid squares (lower is more accurate but slower to build)
grid_x = np.arange(min_x, max_x, resolution)
grid_y = np.arange(min_y, max_y, resolution)

def compute_sdf_matrix(curb_points, grid_x, grid_y, is_left_curb):
    line = LineString(curb_points)
    Z = np.zeros((len(grid_x), len(grid_y)))
    
    print(f"Building SDF Matrix (Left={is_left_curb})... this may take a minute.")
    
    for i, x in enumerate(grid_x):
        for j, y in enumerate(grid_y):
            pt = Point(x, y)
            
            # 1. Absolute distance to the wall
            dist = line.distance(pt)
            
            # 2. Find closest point on the line to determine tangent
            proj_dist = line.project(pt)
            closest_pt = line.interpolate(proj_dist)
            
            # Look slightly ahead on the line to get the tangent vector
            eps = 0.01
            if proj_dist + eps <= line.length:
                ahead_pt = line.interpolate(proj_dist + eps)
                tangent = np.array([ahead_pt.x - closest_pt.x, ahead_pt.y - closest_pt.y])
            else: # Edge case: at the very end of the line, look backward
                behind_pt = line.interpolate(proj_dist - eps)
                tangent = np.array([closest_pt.x - behind_pt.x, closest_pt.y - behind_pt.y])
            
            # Vector from the wall to our grid point
            v = np.array([pt.x - closest_pt.x, pt.y - closest_pt.y])
            
            # 2D Cross Product to determine left/right side
            cross_prod = tangent[0]*v[1] - tangent[1]*v[0]
            
            # Determine Sign (Positive = Inside the track, Negative = Outside)
            if is_left_curb:
                # If it's the left boundary, the safe side is to the RIGHT of the line.
                # A negative cross product means the point is to the right.
                sign = 1.0 if cross_prod < 0 else -1.0
            else:
                # If it's the right boundary, the safe side is to the LEFT of the line.
                sign = 1.0 if cross_prod > 0 else -1.0
                
            Z[i, j] = dist * sign
            
    return Z

path_l, path_r = Path('cached/z_left.npy'), Path('cached/z_right.npy')

if path_l.exists() and path_r.exists():
    print("loading cached SDF data")
    Z_left, Z_right = np.load(path_l), np.load(path_r)
else:
    # Assuming Outer curb is on the Left, Inner curb is on the Right
    Z_left = compute_sdf_matrix(local_outer, grid_x, grid_y, is_left_curb=True)
    Z_right = compute_sdf_matrix(local_inner, grid_x, grid_y, is_left_curb=False)
    
    Path("cached/").mkdir(parents=True, exist_ok=True)
    np.save(path_l, Z_left)
    np.save(path_r, Z_right)

# CRITICAL CASADI DETAIL: Flatten the arrays using Fortran order ('F').
# Because grid_x (rows) varies first in our nested loops, standard 'C' order
# will rotate your map 90 degrees and crash the solver!
flat_Z_left  = Z_left.ravel(order='F')
flat_Z_right = Z_right.ravel(order='F')

# Create the infinitely differentiable B-Splines
d_left_spline  = interpolant('d_left', 'bspline', [grid_x, grid_y], flat_Z_left)
d_right_spline = interpolant('d_right', 'bspline', [grid_x, grid_y], flat_Z_right)

print("Splines generated successfully!")

# Elevation spline
path_elev = Path('cached/elevation.npy')

if path_elev.exists():
    print("loading cached elev grid")
    Z_elev = np.load(path_elev)
else:
    print("Building Elevation Matrix... this may take a minute.")
    emap = EMap("elevation_data/course_cut_square.csv", inner_curb, outer_curb, tolerance=20.0)
    Z_elev = np.zeros((len(grid_x), len(grid_y)))
    
    for i, local_x in enumerate(grid_x):
        for j, local_y in enumerate(grid_y):
            # Shift back to raw UTM to query the EMap correctly
            raw_x = local_x + x0 
            raw_y = local_y + y0
            
            Z_elev[i, j] = emap.elevation(raw_x, raw_y) 
            
    np.save(path_elev, Z_elev)

# Flatten with Fortran order (CRITICAL for CasADi 2D splines)
flat_Z_elev = Z_elev.ravel(order='F')

# Create the infinitely differentiable CasADi B-spline
elevation_spline = interpolant('elev', 'bspline', [grid_x, grid_y], flat_Z_elev)

# Create a dedicated gradient function
# We use a pure dummy symbol just to calculate the analytical derivative once
dummy_xy = MX.sym('dummy_xy', 2)
dummy_z = elevation_spline(dummy_xy)
dummy_grad = jacobian(dummy_z, dummy_xy)

# Bake the derivative into a reusable CasADi Function
grad_elevation_spline = Function('grad_elev', [dummy_xy], [dummy_grad])

print("Elevation Spline Ready!")

opti = Opti()

# decision variables
X = opti.variable(4, N+1)
x = X[0, :]
y = X[1, :]
theta = X[2, :]
v = X[3, :]

U = opti.variable(1, N)
T = opti.variable()

xy_matrix = vertcat(x, y)

dist_L = d_left_spline(xy_matrix)
dist_R = d_right_spline(xy_matrix)

# objective

# --- Objective Weights ---
w_t = 1.0   # Weight for minimizing time
w_c = 10.0   # Weight for penalizing rapid steering
w_b = 30.0  # Weight for penalizing hugging barriers

k_b = 3.0

u_cost = sumsqr(diff(U))
b_cost = sum2(exp(-k_b * (dist_L - 0.1)) + exp(-k_b * (dist_R - 0.1)))

J = w_t * T + w_c * u_cost + w_b * b_cost

opti.minimize(J)

def f(state, u):
    l = wheelbase
    x, y, theta, v = state[0], state[1], state[2], state[3]
    d = u[0]

    g = 9.807
    M = m / (m + (I / (l*l))*(tan(d) ** 2))

    current_xy = vertcat(x, y)

    gradient = grad_elevation_spline(current_xy)
    
    # CasADi jacobian(1x1, 2x1) returns a 1x2 row vector
    dz_dx = gradient[0, 0]
    dz_dy = gradient[0, 1]

    slope_ahead = dz_dx * cos(theta) + dz_dy * sin(theta)

    dv_g = -M * g * slope_ahead

    f_rr = c_rr * m * g
    f_v = c_v * v
    f_a = c_a * v * v
    f_s = c_s * (v ** 2) * (np.sin(d) ** 2)

    dv_f = -M * ((f_rr + f_v + f_a + f_s) / m)


    dv = dv_g + dv_f

    return vertcat(
        v * cos(theta),
        v * sin(theta),
        v / l * tan(d),
        dv
    )

# dynamic constraints
dt = T/N
for k in range(N):
    # rk4 integration
    k1 = f(X[:,k],         U[:,k])
    k2 = f(X[:,k]+dt/2*k1, U[:,k])
    k3 = f(X[:,k]+dt/2*k2, U[:,k])
    k4 = f(X[:,k]+dt*k3,   U[:,k])
    x_next = X[:,k] + dt/6*(k1+2*k2+2*k3+k4) 
    opti.subject_to(X[:,k+1]==x_next)

# steering
d_max = 20 * pi / 180
opti.subject_to(opti.bounded(-d_max, U, d_max))

# path constraints
opti.subject_to(dist_L >= 0.1)
opti.subject_to(dist_R >= 0.1)

# boundary conditions
x1, y1 = 589250.8718076576, 4477264.515107977

x1, y1 = x1 - x0, y1 - y0

opti.subject_to(x[0] == 0.0)
opti.subject_to(y[0] == 0.0)
opti.subject_to(theta[0] == -166.50485229492188 * pi/180)
opti.subject_to(v[0] == 8.0)

opti.subject_to((x[-1] - x1) ** 2 +  (y[-1] - y1) ** 2 <= 0.8)

opti.subject_to(T>=1)

# ==========================================
# INITIAL GUESSES (Loaded from path)
# ==========================================

# Load the saved numpy array
rollout = np.load("cached/rollout_guess.npy")
t_raw = rollout[:, 0]
x_raw = rollout[:, 1]
y_raw = rollout[:, 2]
theta_raw = np.unwrap(rollout[:, 3]) 
v_raw = rollout[:, 4]
u_raw = rollout[:, 5]

# Shift to Local Coordinate Frame
local_x = x_raw - x_raw[0]
local_y = y_raw - y_raw[0]

# Resample to exactly N+1 points
even_times = np.linspace(0, t_raw[-1], N+1)

x_guess = interp1d(t_raw, local_x)(even_times)
y_guess = interp1d(t_raw, local_y)(even_times)
theta_guess = interp1d(t_raw, theta_raw)(even_times)
v_guess = interp1d(t_raw, v_raw)(even_times)

# Steering only needs N points, so we sample slightly differently
even_times_u = np.linspace(0, t_raw[-1], N)
u_guess = interp1d(t_raw, u_raw)(even_times_u)

# Apply to CasADi
opti.set_initial(T, t_raw[-1])
opti.set_initial(x, x_guess)
opti.set_initial(y, y_guess)
opti.set_initial(theta, theta_guess)
opti.set_initial(v, v_guess)
opti.set_initial(U, u_guess)


opti.solver("ipopt")
sol = opti.solve()

# plot
import matplotlib.pyplot as plt

# 1. Extract the numerical values from the solution
opt_x = sol.value(x) + x0
opt_y = sol.value(y) + y0
opt_theta = sol.value(theta)
opt_v = sol.value(v)
opt_u = sol.value(U)
opt_T = sol.value(T)

print(f"Optimal Time: {opt_T:.2f} seconds")

# 2. Plot the Trajectory
plt.figure(figsize=(12, 10))

# --- NEW: Plot the Boundaries ---
# (Assuming local_outer and local_inner are still in memory from the SDF step)
plt.plot(local_outer[:, 0] + x0, local_outer[:, 1] + y0, 'k--', linewidth=1.5, label='Outer Boundary (Left)')
plt.plot(local_inner[:, 0] + x0, local_inner[:, 1] + y0, 'k-', linewidth=1.5, label='Inner Boundary (Right)')

# Plot the CasADi solved path
plt.plot(opt_x, opt_y, 'b.-', linewidth=2, label='Optimal Path')

# Mark start and end points
plt.plot(opt_x[0], opt_y[0], 'go', markersize=8, label='Start')
plt.plot(opt_x[-1], opt_y[-1], 'ro', markersize=8, label='Finish')

# Draw the target finish radius (Now using the Local coordinates)
target_circle = plt.Circle((x1 + x0, y1 + y0), np.sqrt(0.8), 
                           color='r', fill=False, linestyle='--', label='Target Area')
plt.gca().add_patch(target_circle)

plt.xlabel('Local X (m)')
plt.ylabel('Local Y (m)')
plt.title('Optimal Buggy Trajectory with Boundaries')
plt.legend()
plt.axis('equal') # CRITICAL: Keeps X and Y scale 1:1 so corners aren't distorted
plt.grid(True)

# 3. Plot the Steering Inputs
plt.figure(figsize=(10, 4))
plt.plot(np.linspace(0, opt_T, N), opt_u * (180/np.pi), 'r.-')
plt.xlabel('Time (s)')
plt.ylabel('Steering Angle (Degrees)')
plt.title('Optimal Steering Control')
plt.grid(True)

# Show both plots
plt.show()

# ==========================================
# EXPORT OPTIMIZED TRAJECTORY
# ==========================================
print("Exporting trajectory to JSON...")

# 1. Shift local CasADi coordinates back to raw UTM

waypoints = []

for x, y in zip(opt_x, opt_y):
    # Convert UTM (Easting, Northing) -> (Latitude, Longitude)
    # Schenley Park is UTM Zone 17, Band 'T' (Northern Hemisphere)
    lat, lon = utm.to_latlon(x, y, Constants.UTM_ZONE_NUM, Constants.UTM_ZONE_LETTER)
    
    waypoints.append({
        "key": str(uuid.uuid4()),
        "lat": float(lat),
        "lon": float(lon),
        "active": False
    })

# 2. Save to file
output_path = "paths/freeroll_optim.json"
with open(output_path, "w") as f:
    json.dump(waypoints, f, indent=2)

print(f"Successfully saved {len(waypoints)} waypoints to {output_path}!")
