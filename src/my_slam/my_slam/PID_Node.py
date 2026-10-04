import rclpy
import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Int32
from std_msgs.msg import Float64MultiArray

p = 0.25

class MyNode(Node):
    def __init__(self):
        super().__init__("TestNode")

        self.currentLocation = None
        self.newLocation = None
        self.newTarget = None

        self.stateVector = np.array([0, # x position
                                             0, # y position
                                             0, # rotation
                                             0, # linear velocity
                                             0, # angular velocity
                                             0, # linear bias
                                             0]) # angular bias

        self.get_logger().info("Test Node has Started!")

        self.publisher = self.create_publisher(
            TwistStamped,
            '/cmd_vel',
            10
        )

        self.EKF_subscriber = self.create_subscription(
            Float64MultiArray,
            '/corrected',
            self.callBackCurrent,
            10
        )

        self.target_subscriber = self.create_subscription(
            Int32,
            '/targetPosition',
            self.callBackTarget,
            10
        )

    def callBackCurrent(self, mes):
        print("TRYING TO UPDATE CURRENT")

        self.stateVector = np.array(mes.data)[:3]

        self.updatePID()

    def callBackTarget(self, mes):

        self.newTarget = mes.data

        print("UPDATED TARGET")

        self.updatePID()

    def updatePID(self):
        if self.newLocation is None:
            return

        if self.newTarget is None:
            return

        print("UPDATING PID")

        error = self.newTarget - self.newLocation

        self.xVelocity = p * error

        self.publish_command()

    def publish_command(self, linVel, angVel):

        msg = TwistStamped()

        msg.header.stamp = self.get_clock().now().to_msg()

        msg.twist.linear.x = linVel
        msg.twist.angular.z = angVel

        self.publisher.publish(msg)

def main(args = None):
    rclpy.init(args=args)

    newNode = MyNode()

    rclpy.spin(newNode)

    newNode.destroy_node()

    rclpy.shutdown()

if __name__ == "__main__":
    main()

