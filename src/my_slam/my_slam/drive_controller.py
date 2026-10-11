import rclpy
import numpy as np

from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import TwistStamped

class driveController(Node):
    def __init__(self):
        super().__init__("driveController")

        self.stateVector = np.zeros(7)
        self.velocities = []
        self.pointList = []
        self.previousIndex = 0
        self.haveState = False
        self.MAXVELOCITYANGULAR = 0.3

        self.state_subscriber = self.create_subscription (
            Float64MultiArray,
            '/prediction',
            self.updateState,
            1
        )

        self.velocity_subscriber = self.create_subscription (
            Float64MultiArray,
            '/velocity',
            self.updateVelocities,
            1
        )

        self.spline_subscriber = self.create_subscription (
            Float64MultiArray,
            '/smoothedPath',
            self.updateSpline,
            1
        )

        self.cmd_vel_publisher = self.create_publisher(
            TwistStamped,
            '/cmd_vel',
            10
        )

        self.control_timer = self.create_timer(
            1.0 / 30.0,
            self.controlLoop
        )

    def updateState(self, message):
        if self.haveState == False:
            print("UPDATED STATE")
        self.stateVector = np.array(message.data)
        self.haveState = True

    def updateVelocities(self, message):
        if self.velocities == []:
            print("UPDATING VELOCITIES")
        self.velocities = message.data
        self.previousIndex = 0

    def updateSpline(self, message):
        if self.pointList == []:
            print("UPDATING SPLINE")
        points = []
        x = 0

        for i, value in enumerate(message.data):
            if i % 2 == 0:
                x = value
            else:
                points.append([x, value])

        self.pointList = points

    def locateClosestPoint(self):
        closestIndex = -1
        closestDistance = np.inf
        
        for i, point in enumerate(self.pointList[self.previousIndex:], start=self.previousIndex):
            distance = np.sqrt((point[0] - self.stateVector[0]) ** 2 + (point[1] - self.stateVector[1]) ** 2)

            if distance <= closestDistance:
                closestDistance = distance
                closestIndex = i

        self.previousIndex = closestIndex

    def findTargetPoint(self):
        currentDistance = 0
        previousPoint = self.pointList[self.previousIndex]

        for point in self.pointList[self.previousIndex:]:
            distance = np.sqrt((previousPoint[0] - point[0]) ** 2 +(previousPoint[1] - point[1]) ** 2)
            currentDistance += distance
            previousPoint = point
            
            if currentDistance >= 0.3:
                return point

        return previousPoint
                
    def calculateAngularVel(self, targetPoint, linearVelocity):
        deltaX = targetPoint[0] - self.stateVector[0]
        deltaY = targetPoint[1] - self.stateVector[1]
        theta = self.stateVector[2]

        x = np.cos(theta) * deltaX + np.sin(theta) * deltaY
        y = -np.sin(theta) * deltaX + np.cos(theta) * deltaY

        L = x ** 2 + y ** 2
        if L < 1e-8:
            return 0.0
        
        kpp = 2 * y / L
        return linearVelocity * kpp

    def controlLoop(self):
        if self.velocities != [] and self.haveState and self.pointList != []:
            print("WE DRIVING")
            self.locateClosestPoint()
            linearVelocity = self.velocities[self.previousIndex]
            targetPoint = self.findTargetPoint()
            angularVelocity = min(self.MAXVELOCITYANGULAR, self.calculateAngularVel(targetPoint, linearVelocity))


            msg = TwistStamped()
            
            msg.header.stamp = self.get_clock().now().to_msg()

            msg.twist.linear.x = linearVelocity
            msg.twist.angular.z = angularVelocity

            self.cmd_vel_publisher.publish(msg)

def main(args = None):
    print("WE DRIVE CONTROLLING")

    rclpy.init(args = args)

    driveNode = driveController()

    rclpy.spin(driveNode)

    driveNode.destroy_node()

    rclpy.shutdown()

if __name__ == '__main__':
    main()

