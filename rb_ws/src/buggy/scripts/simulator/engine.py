#! /usr/bin/env python3
import json
import os
import threading
import time
from collections import deque
import rclpy
from buggy.msg import StampedFloat64Msg
from rclpy.node import Node
from geometry_msgs.msg import Pose, Twist, PoseWithCovariance, TwistWithCovariance
from std_msgs.msg import Float64, Bool
from sensor_msgs.msg import NavSatFix
from nav_msgs.msg import Odometry
import numpy as np
import utm
from util.constants import Constants
from util.emap import EMap


class Simulator(Node):

    def __init__(self):
        super().__init__('engine')
        self.get_logger().info('INITIALIZED.')

        self.starting_poses = {
            "Hill1_NAND": (589760.46, 4477322.07, -110),
            "Hill1_SC": (589761.40, 4477321.75, -110),
            "Hill2_NAND": (Constants.UTM_EAST_ZERO + 20, Constants.UTM_NORTH_ZERO + 30, -110),
            "Hill2_SC": (Constants.UTM_EAST_ZERO + 20, Constants.UTM_NORTH_ZERO + 30, -110),
            "WESTINGHOUSE": (589647, 4477143, -150),
            "UC_TO_PURNELL": (589635, 4477468, 160),
            "UC": (589681, 4477457, 160),
            "TRACK_EAST_END": (589953, 4477465, 90),
            "TRACK_RESNICK": (589906, 4477437, -20),
            "GARAGE": (589846, 4477580, 180),
            "FREW_ST": (589646, 4477359, -20),
            "FREW_ST_PASS": (589644, 4477368, -20),
            "RACELINE_PASS": (589468.02, 4476993.07, -160),
        }

        if (self.get_namespace() == "/SC"):
            self.buggy_name = "SC"
            self.declare_parameter('pose', "Hill1_SC")
            self.wheelbase = Constants.WHEELBASE_SC

        elif (self.get_namespace() == "/NAND"):
            self.buggy_name = "NAND"
            self.declare_parameter('pose', "Hill1_NAND")
            self.wheelbase = Constants.WHEELBASE_NAND

        self.declare_parameter('velocity', 12.0)
        self.velocity = self.get_parameter("velocity").value

        self.is_freeroll = False

        self.declare_parameter('steering_offset', 0.0)
        self.declare_parameter('steering_offset_func', 'constant')
        self.declare_parameter('steering_offset_period', 10.0)
        self.declare_parameter('edata', 'course_cut_square.csv')

        # Physical constants
        # TODO - set as params and/or move to constants
        self.mass = 58.967  # kg
        self.moment_of_inertia = 21.847274476  # kg, m, ellipsoid + ballpark measurement

        # TODO - params to specify boundary path
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
        
        course_name = "paths/buggycourse"
        outer_file = f"{course_name}_outer.json"
        inner_file = f"{course_name}_inner.json"

        outer_points = load_json_to_utm(outer_file)
        inner_points = load_json_to_utm(inner_file)

        emap_name = self.get_parameter("edata").value
        emap_path = os.environ["EDATAPATH"] + emap_name
        self.emap = EMap(emap_path, inner_points, outer_points)

        self.steering_offset = self.get_parameter("steering_offset").value
        self.steering_offset_func = str(self.get_parameter("steering_offset_func").value).lower()
        self.steering_offset_period = self.get_parameter("steering_offset_period").value

        if self.steering_offset_func not in {"constant", "sin"}:
            self.get_logger().warning(
                f"Unknown steering_offset_func '{self.steering_offset_func}'. Falling back to 'constant'."
            )
            self.steering_offset_func = "constant"

        if self.steering_offset_func == "sin" and self.steering_offset_period <= 0:
            self.get_logger().warning(
                "steering_offset_period must be positive. Using default of 10s."
            )
            self.steering_offset_period = 10.0

        self.sim_time = 0.0

        self.declare_parameter("process_noise_std", 0.0)
        self.process_noise_std = self.get_parameter("process_noise_std").value
        self.declare_parameter("measurement_noise_std", 1e-2)
        self.measurement_noise_std = self.get_parameter("measurement_noise_std").value

        init_pose_name = self.get_parameter("pose").value
        self.init_pose = self.starting_poses[init_pose_name]

        self.e_utm, self.n_utm, self.heading = self.init_pose
        self.current_steering = 0.0  # degrees
        self.rate = 100  # Hz
        self.tick_count = 0
        self.interval = 2  # how frequently to publish

        # Steering delay configuration (each step = 10ms at 100 Hz)
        self.declare_parameter("steering_delay", 0)
        self.steering_delay_steps = self.get_parameter("steering_delay").value
        self.get_logger().info(
            f"Steering delay set to {self.steering_delay_steps} steps."
        )

        # Use deque as a delay line - current steering is at the end, delayed at the front
        self.steering_buffer = deque(maxlen=max(1, self.steering_delay_steps + 1))
        # Initialize buffer with zero steering commands
        for _ in range(self.steering_buffer.maxlen):
            self.steering_buffer.append(0.0)

        self.lock = threading.Lock()

        timer_period = 1/self.rate  # seconds
        self.timer = self.create_timer(timer_period, self.loop)

        self.steering_subscriber = self.create_subscription(
            StampedFloat64Msg, "input/steering", self.update_steering_angle, 1
        )

        # To read from velocity
        self.velocity_subscriber = self.create_subscription(
            Float64, "sim/velocity", self.update_velocity, 1
        )

        self.simstate_subscriber = self.create_subscription(
            Bool, "sim/freeroll", self.update_simstate, 1
        )

        # simulate the INS's outputs (noise included)
        # this is published as a BuggyState (UTM and radians)
        self.pose_publisher = self.create_publisher(Odometry, "self/state", 1)

        self.navsatfix_noisy_publisher = self.create_publisher(
                NavSatFix, "self/pose_navsat_noisy", 1
        )
        self.offset_publisher = self.create_publisher(
                Float64, "sim/true_offset", 1
        )

    def update_steering_angle(self, data: StampedFloat64Msg):
        with self.lock:
            # add new steering command to buffer
            self.steering_buffer.append(data.data)

    def apply_delayed_steering(self):
        """Precondition: lock must be held when calling this function"""
        # the delayed steering is at the front of the buffer
        cur_steer = self.current_steering
        new_steer = self.steering_buffer[0]

        self.current_steering = new_steer
        if not self.is_freeroll:
            return

        offset = self.steering_offset_value(self.sim_time)

        # In freeroll, adjust velocity to preserve kinetic energy
        def f(d):
            # v^2 factor
            return self.mass + (self.moment_of_inertia / self.wheelbase ** 2) * (np.tan(d) ** 2)

        v = self.velocity
        d = (cur_steer + offset) * np.pi / 180.0
        E = v*v*f(d)

        d = (new_steer + offset) * np.pi / 180
        self.velocity = np.sqrt(E / f(d))

    def steering_offset_value(self, sim_time: float) -> float:
        if self.steering_offset_func == "sin":
            phase = 2 * np.pi * sim_time / self.steering_offset_period
            return self.steering_offset * np.sin(phase)
        return self.steering_offset

    def update_velocity(self, data: Float64):
        with self.lock:
            # Ignore velocity updates while in freeroll
            if not self.is_freeroll:
                self.velocity = data.data
    
    def update_simstate(self, is_freeroll: Bool):
        with self.lock:
            self.is_freeroll = is_freeroll.data

    def dynamics(self, state):
        l = self.wheelbase
        m = self.mass
        I = self.moment_of_inertia
        x, y, theta, v, delta, delta0 = state

        d = delta + delta0

        dv = 0

        if self.is_freeroll:
            # gravity - assumes CoM is in back axle for now
            g = 9.807
            M = m / (m + (I / (l*l))*(np.tan(d) ** 2))
            t = np.array([np.cos(theta), np.sin(theta)])
            dv = -M * g * np.dot(t, self.emap.grad(x, y))

        return np.array([v * np.cos(theta),
                         v * np.sin(theta),
                         v / l * np.tan(d),
                         dv,
                         0, 0])

    def step(self):
        with self.lock:
            heading = self.heading
            e_utm = self.e_utm
            n_utm = self.n_utm

            self.apply_delayed_steering()
            velocity = self.velocity
            steering_angle = self.current_steering
            sim_time = self.sim_time
            steering_offset_deg = self.steering_offset_value(sim_time)
            self.offset_publisher.publish(Float64(data=steering_offset_deg))

        h = 1/self.rate
        state = np.array([
            e_utm,
            n_utm,
            np.deg2rad(heading),
            velocity,
            np.deg2rad(steering_angle),
            np.deg2rad(steering_offset_deg),
        ])
        k1 = self.dynamics(state)
        k2 = self.dynamics(state + h/2 * k1)
        k3 = self.dynamics(state + h/2 * k2)
        k4 = self.dynamics(state + h * k3)

        final_state = state + h/6 * (k1 + 2 * k2 + 2 * k3 + k4)
        final_state[0] += np.random.normal(0, self.process_noise_std)
        final_state[1] += np.random.normal(0, self.process_noise_std)

        e_utm_new, n_utm_new, heading_new, velocity_new, _, _ = final_state
        heading_new = np.rad2deg(heading_new)

        with self.lock:
            self.e_utm = e_utm_new
            self.n_utm = n_utm_new
            self.heading = heading_new
            self.velocity = velocity_new
            self.sim_time = sim_time + h

    def publish(self):
        odom_pose = Pose()
        time_stamp = self.get_clock().now().to_msg()
        with self.lock:
            odom_pose.position.x = self.e_utm
            odom_pose.position.y = self.n_utm
            velocity = self.velocity

        odom_pose.position.x += np.random.normal(0, self.measurement_noise_std)
        odom_pose.position.y += np.random.normal(0, self.measurement_noise_std)

        (lat, long) = utm.to_latlon(
            odom_pose.position.x,
            odom_pose.position.y,
            Constants.UTM_ZONE_NUM,
            Constants.UTM_ZONE_LETTER,
        )

        nsf_noisy = NavSatFix()
        nsf_noisy.latitude = lat
        nsf_noisy.longitude = long
        nsf_noisy.header.stamp = time_stamp
        self.navsatfix_noisy_publisher.publish(nsf_noisy)

        odom = Odometry()
        odom.header.stamp = time_stamp

        odom_pose.position.z = float(self.emap.elevation(odom_pose.position.x, odom_pose.position.y))
        odom_pose.orientation.z = np.deg2rad(self.heading)

        # variance on x and y from measurement_noise_std
        odom_pose_covariance = [0.0] * 36
        measure_noise_var = self.measurement_noise_std ** 2
        odom_pose_covariance[0] = measure_noise_var   # x
        odom_pose_covariance[7] = measure_noise_var   # y

        # NOTE: autonsystem only cares about magnitude of velocity, so we don't need to split into components
        odom_twist = Twist()
        odom_twist.linear.x = float(velocity)

        odom.pose = PoseWithCovariance(pose=odom_pose, covariance=odom_pose_covariance)
        odom.twist = TwistWithCovariance(twist=odom_twist)

        ns = time.time_ns()
        odom.header.stamp.sec = ((ns // int(1e9)) + 2**31) % 2**32 - 2**31
        odom.header.stamp.nanosec = ns % int(1e9)

        self.pose_publisher.publish(odom)

    def loop(self):
        self.step()
        if self.tick_count % self.interval == 0:
            self.publish()
        self.tick_count += 1
        self.get_logger().debug(
            "SIMULATED UTM: ({}, {}), HEADING: {}".format(
                self.e_utm, self.n_utm, self.heading
            )
        )


def main(args=None):
    rclpy.init(args=args)
    sim = Simulator()
    for _ in range(500):
        time.sleep(0.01)
        sim.publish()

    sim.get_logger().info("STARTED PUBLISHING")
    rclpy.spin(sim)

    sim.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
