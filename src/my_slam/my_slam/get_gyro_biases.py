import rclpy
import time
import numpy as np

from rclpy.node import Node
from sensor_msgs.msg import Imu

class gyro_biases(Node):
    def __init__(self):
        super().__init__("get_Gyro_Biases")

        self.gyroRotationBiasZ = 0
        self.gyroLinearBias = 0

        self.startTime = time.perf_counter()
        self.counter = 0

        self.IMU_subscriber = self.create_subscription(
            Imu,
            '/imu',
            self.updateAverage,
            1
        )


    def updateAverage(self, msg):
        self.counter += 1
        self.gyroRotationBiasZ += msg.angular_velocity.z

        self.gyroLinearBias += msg.linear_acceleration.x
        self.gyroLinearBias += msg.linear_acceleration.y

        if time.perf_counter() - self.startTime > 30:
            print("RotationBias: ", self.gyroRotationBiasZ / self.counter)
            print("LinearBias: ", self.gyroLinearBias / self.counter)

            self.startTime = time.perf_counter()


def main(args = None):
    rclpy.init(args=args)

    newNode = gyro_biases()

    rclpy.spin(newNode)

    newNode.destroy_node()

    rclpy.shutdown()

if __name__ == "__main__":
    main()