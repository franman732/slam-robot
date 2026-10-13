import rclpy
import numpy as np

from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Bool

class driveController(Node):
    def __init__(self):
        super().__init__("driveController")

        self.stateVector = np.zeros(7)
        self.velocities = []
        self.pointList = []
        self.previousIndex = 0
        self.haveState = False

        self.MAXVELOCITYANGULAR = 0.3
        self.SEARCHWINDOW = 300 # How many spline points ahead of previousIndex we look at for the closest point. This stops the index from jumping if the path ever passes close to itself.
        self.MINLOOKAHEAD = 0.2 # Was a fixed 0.3. A shorter lookahead cuts corners less, but raise it again if the car starts to wobble on straight sections.
        self.LOOKAHEADGAIN = 1.0 # Lookahead distance = LOOKAHEADGAIN * linearVelocity, but never less than MINLOOKAHEAD.
        self.GOALTOLERANCE = 0.05
        self.ENDDISTANCE = 0.06 # We move to the next path if the distance from the robot to the end square is less than .06.

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
            '/correctPath',
            self.updateSpline,
            1
        )

        self.cmd_vel_publisher = self.create_publisher(
            TwistStamped,
            '/cmd_vel',
            10
        )

        self.update_path_publisher = self.create_publisher (
            Bool,
            '/updatePath',
            1
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
        self.velocities = list(message.data)
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
        self.previousIndex = 0

    def locateClosestPoint(self):
        closestIndex = self.previousIndex
        closestDistance = np.inf

        endIndex = min(len(self.pointList), self.previousIndex + self.SEARCHWINDOW)
        
        for i in range(self.previousIndex, endIndex):
            point = self.pointList[i]
            distance = np.sqrt((point[0] - self.stateVector[0]) ** 2 + (point[1] - self.stateVector[1]) ** 2)

            if distance <= closestDistance:
                closestDistance = distance
                closestIndex = i

        self.previousIndex = closestIndex

    def findTargetPoint(self, lookahead):
        currentDistance = 0
        previousPoint = self.pointList[self.previousIndex]

        for point in self.pointList[self.previousIndex:]:
            distance = np.sqrt((previousPoint[0] - point[0]) ** 2 +(previousPoint[1] - point[1]) ** 2)
            currentDistance += distance
            previousPoint = point
            
            if currentDistance >= lookahead:
                return point

        return previousPoint

    def getTargetInRobotFrame(self, targetPoint): # x is how far in front of the car the target is, y is how far to the left.
        deltaX = targetPoint[0] - self.stateVector[0]
        deltaY = targetPoint[1] - self.stateVector[1]
        theta = self.stateVector[2]

        x = np.cos(theta) * deltaX + np.sin(theta) * deltaY
        y = -np.sin(theta) * deltaX + np.cos(theta) * deltaY

        return x, y
                
    def calculateAngularVel(self, x, y, linearVelocity):
        L = x ** 2 + y ** 2
        if L < 1e-8:
            return 0.0
        
        kpp = 2 * y / L
        return linearVelocity * kpp

    def publishVelocity(self, linearVelocity, angularVelocity):
        msg = TwistStamped()
        
        msg.header.stamp = self.get_clock().now().to_msg()

        msg.twist.linear.x = float(linearVelocity)
        msg.twist.angular.z = float(angularVelocity)

        print("FORWARD VELOCITY: ", linearVelocity)
        print("ANGULAR VELOCITY: ", angularVelocity)

        self.cmd_vel_publisher.publish(msg)

    def controlLoop(self):
        if len(self.velocities) == 0 or not self.haveState or len(self.pointList) == 0:
            return

        if len(self.velocities) != len(self.pointList): # /velocity and /correctPath arrive separately, so wait until both belong to the same path.
            print("VELOCITIES DO NOT MATCH THE SPLINE YET")
            return

        self.locateClosestPoint()

        linearVelocity = self.velocities[self.previousIndex]

        lookahead = max(self.MINLOOKAHEAD, self.LOOKAHEADGAIN * linearVelocity)
        targetPoint = self.findTargetPoint(lookahead)

        x, y = self.getTargetInRobotFrame(targetPoint)

        if x <= 0: # The target is behind the car, where pure pursuit doesn't mean anything, so turn in place towards it instead.
            linearVelocity = 0.0

            if y >= 0:
                angularVelocity = self.MAXVELOCITYANGULAR
            else:
                angularVelocity = -self.MAXVELOCITYANGULAR
        else:
            angularVelocity = self.calculateAngularVel(x, y, linearVelocity)

            if abs(angularVelocity) > self.MAXVELOCITYANGULAR: # Slows the car down by the same amount instead of only clamping, so the car still follows the same curve.
                linearVelocity = linearVelocity * self.MAXVELOCITYANGULAR / abs(angularVelocity)
                angularVelocity = np.sign(angularVelocity) * self.MAXVELOCITYANGULAR

        self.publishVelocity(linearVelocity, angularVelocity)

        if np.sqrt((self.pointList[-1][0] - self.stateVector[0]) ** 2 + (self.pointList[-1][1] - self.stateVector[1]) ** 2) < self.ENDDISTANCE:
            msg = Bool()
            msg.data = True
            
            self.update_path_publisher.publish(msg)
            print("NEW PATH")
        else:
            print("DISTANCE: ", np.sqrt((self.pointList[-1][0] - self.stateVector[0]) ** 2 + (self.pointList[-1][1] - self.stateVector[1]) ** 2))

def main(args = None):
    print("WE DRIVE CONTROLLING")

    rclpy.init(args = args)

    driveNode = driveController()

    rclpy.spin(driveNode)

    driveNode.destroy_node()

    rclpy.shutdown()

if __name__ == '__main__':
    main()