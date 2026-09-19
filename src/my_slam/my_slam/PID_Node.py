import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from tf2_msgs.msg import TFMessage
from std_msgs.msg import Int32

p = 0.25

class MyNode(Node):
    def __init__(self):
        super().__init__("TestNode")

        self.currentLocation = None
        self.newLocation = None
        self.newTarget = None

        self.xVelocity = 0.0

        self.get_logger().info("Test Node has Started!")

        self.publisher = self.create_publisher(
            TwistStamped,
            '/cmd_vel',
            10
        )

        self.tf_subscriber = self.create_subscription(
            TFMessage,
            '/tf',
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

        for transform in mes.transforms:

            if transform.child_frame_id == 'base_footprint':
                print("UPDATING CURRENT")
                
                self.newLocation = transform.transform.translation.x
                break

        self.updatePID()

    def callBackTarget(self, mes):

        self.newTarget = mes.data

        print("UPDATED TARGET")

        self.updatePID()

    def updatePID(self):

        # We cannot calculate anything until we
        # have both pieces of information.

        if self.newLocation is None:
            return

        if self.newTarget is None:
            return

        print("UPDATING PID")

        # Calculate error
        error = self.newTarget - self.newLocation

        # Proportional controller
        self.xVelocity = p * error

        print(f"Current: {self.newLocation}")
        print(f"Target: {self.newTarget}")
        print(f"Error: {error}")
        print(f"Velocity: {self.xVelocity}")

        self.publish_command()

    def publish_command(self):

        msg = TwistStamped()

        msg.header.stamp = self.get_clock().now().to_msg()

        msg.twist.linear.x = self.xVelocity
        msg.twist.angular.z = 0.0

        self.publisher.publish(msg)

def main(args = None):
    rclpy.init(args=args)

    newNode = MyNode()

    rclpy.spin(newNode)

    newNode.destroy_node()

    rclpy.shutdown()

if __name__ == "__main__":
    main()

