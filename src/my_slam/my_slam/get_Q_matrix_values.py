#!/usr/bin/env python3

import csv
import math
import os
import time
from datetime import datetime

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan


class QStationaryTest(Node):

    def __init__(self):
        super().__init__('q_stationary_test')

        # ------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------

        self.declare_parameter('duration', 300.0)
        self.declare_parameter(
            'output_directory',
            os.path.expanduser('~/slamOne/q_data')
        )

        self.duration = float(
            self.get_parameter('duration').value
        )

        self.output_directory = (
            self.get_parameter('output_directory').value
        )

        os.makedirs(self.output_directory, exist_ok=True)

        # ------------------------------------------------------------
        # Timing
        # ------------------------------------------------------------

        self.start_time = None

        self.imu_count = 0
        self.odom_count = 0
        self.scan_count = 0

        self.last_imu_time = None
        self.last_odom_time = None
        self.last_scan_time = None

        # ------------------------------------------------------------
        # Latest sensor values
        # ------------------------------------------------------------

        self.latest_imu = None
        self.latest_odom = None
        self.latest_scan = None

        # ------------------------------------------------------------
        # CSV
        # ------------------------------------------------------------

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        self.csv_path = os.path.join(
            self.output_directory,
            f'stationary_test_{timestamp}.csv'
        )

        self.csv_file = open(
            self.csv_path,
            'w',
            newline=''
        )

        self.csv_writer = csv.writer(self.csv_file)

        self.csv_writer.writerow([
            'wall_time',

            # IMU timing
            'imu_time',
            'imu_dt',

            # IMU angular velocity
            'gyro_x',
            'gyro_y',
            'gyro_z',

            # IMU acceleration
            'accel_x',
            'accel_y',
            'accel_z',

            # IMU orientation
            'orientation_x',
            'orientation_y',
            'orientation_z',
            'orientation_w',

            # Odom timing
            'odom_time',
            'odom_dt',

            # Odom pose
            'odom_x',
            'odom_y',
            'odom_z',

            # Odom orientation
            'odom_qx',
            'odom_qy',
            'odom_qz',
            'odom_qw',

            # Odom velocity
            'odom_vx',
            'odom_vy',
            'odom_vz',

            # Odom angular velocity
            'odom_wx',
            'odom_wy',
            'odom_wz',

            # Scan timing
            'scan_time',
            'scan_dt'
        ])

        # ------------------------------------------------------------
        # Subscriptions
        # ------------------------------------------------------------

        self.imu_sub = self.create_subscription(
            Imu,
            '/imu',
            self.imu_callback,
            100
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            100
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            20
        )

        # ------------------------------------------------------------
        # Timer
        # ------------------------------------------------------------

        self.timer = self.create_timer(
            0.01,
            self.timer_callback
        )

        self.get_logger().info('')
        self.get_logger().info(
            '=================================================='
        )
        self.get_logger().info(
            '        EKF Q MATRIX STATIONARY TEST'
        )
        self.get_logger().info(
            '=================================================='
        )
        self.get_logger().info(
            f'Duration: {self.duration:.1f} seconds'
        )
        self.get_logger().info(
            f'Output: {self.csv_path}'
        )
        self.get_logger().info('')
        self.get_logger().info(
            'MAKE SURE THE ROBOT IS COMPLETELY STATIONARY.'
        )
        self.get_logger().info(
            'Starting data collection...'
        )

    # ================================================================
    # Time conversion
    # ================================================================

    @staticmethod
    def stamp_to_seconds(stamp):
        return (
            float(stamp.sec)
            + float(stamp.nanosec) * 1e-9
        )

    # ================================================================
    # IMU callback
    # ================================================================

    def imu_callback(self, msg):

        current_time = self.stamp_to_seconds(
            msg.header.stamp
        )

        if self.start_time is None:
            self.start_time = time.time()

        if self.last_imu_time is None:
            imu_dt = 0.0
        else:
            imu_dt = current_time - self.last_imu_time

        self.last_imu_time = current_time

        self.latest_imu = {
            'time': current_time,
            'dt': imu_dt,

            'gx': msg.angular_velocity.x,
            'gy': msg.angular_velocity.y,
            'gz': msg.angular_velocity.z,

            'ax': msg.linear_acceleration.x,
            'ay': msg.linear_acceleration.y,
            'az': msg.linear_acceleration.z,

            'qx': msg.orientation.x,
            'qy': msg.orientation.y,
            'qz': msg.orientation.z,
            'qw': msg.orientation.w
        }

        self.imu_count += 1

    # ================================================================
    # Odometry callback
    # ================================================================

    def odom_callback(self, msg):

        current_time = self.stamp_to_seconds(
            msg.header.stamp
        )

        if self.start_time is None:
            self.start_time = time.time()

        if self.last_odom_time is None:
            odom_dt = 0.0
        else:
            odom_dt = current_time - self.last_odom_time

        self.last_odom_time = current_time

        self.latest_odom = {
            'time': current_time,
            'dt': odom_dt,

            'x': msg.pose.pose.position.x,
            'y': msg.pose.pose.position.y,
            'z': msg.pose.pose.position.z,

            'qx': msg.pose.pose.orientation.x,
            'qy': msg.pose.pose.orientation.y,
            'qz': msg.pose.pose.orientation.z,
            'qw': msg.pose.pose.orientation.w,

            'vx': msg.twist.twist.linear.x,
            'vy': msg.twist.twist.linear.y,
            'vz': msg.twist.twist.linear.z,

            'wx': msg.twist.twist.angular.x,
            'wy': msg.twist.twist.angular.y,
            'wz': msg.twist.twist.angular.z
        }

        self.odom_count += 1

    # ================================================================
    # LaserScan callback
    # ================================================================

    def scan_callback(self, msg):

        current_time = self.stamp_to_seconds(
            msg.header.stamp
        )

        if self.last_scan_time is None:
            scan_dt = 0.0
        else:
            scan_dt = current_time - self.last_scan_time

        self.last_scan_time = current_time

        self.latest_scan = {
            'time': current_time,
            'dt': scan_dt
        }

        self.scan_count += 1

    # ================================================================
    # Timer
    # ================================================================

    def timer_callback(self):

        if self.start_time is None:
            return

        elapsed = time.time() - self.start_time

        # ------------------------------------------------------------
        # Write synchronized latest data
        # ------------------------------------------------------------

        imu = self.latest_imu
        odom = self.latest_odom
        scan = self.latest_scan

        # We need both IMU and odom before writing a useful row.
        if imu is not None and odom is not None:

            if scan is not None:
                scan_time = scan['time']
                scan_dt = scan['dt']
            else:
                scan_time = ''
                scan_dt = ''

            self.csv_writer.writerow([
                time.time(),

                imu['time'],
                imu['dt'],

                imu['gx'],
                imu['gy'],
                imu['gz'],

                imu['ax'],
                imu['ay'],
                imu['az'],

                imu['qx'],
                imu['qy'],
                imu['qz'],
                imu['qw'],

                odom['time'],
                odom['dt'],

                odom['x'],
                odom['y'],
                odom['z'],

                odom['qx'],
                odom['qy'],
                odom['qz'],
                odom['qw'],

                odom['vx'],
                odom['vy'],
                odom['vz'],

                odom['wx'],
                odom['wy'],
                odom['wz'],

                scan_time,
                scan_dt
            ])

        # ------------------------------------------------------------
        # Flush periodically so data isn't lost
        # ------------------------------------------------------------

        if int(elapsed) % 5 == 0:
            self.csv_file.flush()

        # ------------------------------------------------------------
        # Progress
        # ------------------------------------------------------------

        if int(elapsed) % 10 == 0:

            self.get_logger().info(
                f'Elapsed: {elapsed:7.1f} / {self.duration:.1f} s | '
                f'IMU: {self.imu_count:7d} | '
                f'Odom: {self.odom_count:7d} | '
                f'Scan: {self.scan_count:7d}'
            )

        # ------------------------------------------------------------
        # Finish
        # ------------------------------------------------------------

        if elapsed >= self.duration:

            self.get_logger().info('')
            self.get_logger().info(
                '=================================================='
            )
            self.get_logger().info(
                'Stationary test complete.'
            )
            self.get_logger().info(
                f'IMU messages:  {self.imu_count}'
            )
            self.get_logger().info(
                f'Odom messages: {self.odom_count}'
            )
            self.get_logger().info(
                f'Scan messages: {self.scan_count}'
            )
            self.get_logger().info(
                f'Data saved to: {self.csv_path}'
            )
            self.get_logger().info(
                '=================================================='
            )

            self.csv_file.flush()
            self.csv_file.close()

            rclpy.shutdown()


def main(args=None):

    rclpy.init(args=args)

    node = QStationaryTest()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        if not node.csv_file.closed:
            node.csv_file.flush()
            node.csv_file.close()

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()