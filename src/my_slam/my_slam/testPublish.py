import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from tf2_msgs.msg import TFMessage
from std_msgs.msg import Int32

p = 0.25

class MyNode(Node):
    def __init__(self):
        super().__init__("TestNode")

        self.get_logger().info("Test Node has Started!")

        self.tf_subscriber = self.create_subscription(
            TFMessage,
            '/tf',
            self.callBack,
            10
        )

    def callBack(self, msg):
        print("MESSAGE: ", msg)
        for transform in msg.transforms:
            print("FIRST: ", transform)

            if transform.child_frame_id == 'base_footprint':
                print("SECOND: ", transform.transform)
                print("THIRD: ", transform.transform.translation)
                print("FINAL: ", transform.transform.translation.x)

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

