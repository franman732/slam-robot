import rclpy
import numpy as np

from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

class calculateLinearVelocities(Node):
    def __init__(self):
        super().__init__("linearVelocityCalculation")

        self.derivativesList = []
        self.pointsList = []
        self.pendingMessage = None # A path that showed up before its matching waypoints and derivatives. It gets calculated as soon as they arrive.

        self.MAXACCELERATION = 0.2
        self.MAXVELOCITY = 0.5
        self.MAXVELOCITYANGULAR = 0.3
        self.MAXBREAK = 0.1
        self.STARTVELOCITY = 0.2 # The controller reads the velocity under the car at the start, so this can't be 0 or the car never leaves the start.
        self.MINVELOCITY = 0.05 # Stops a sharp corner from asking for a velocity so small that the car stalls in the middle of the path.

        self.smoothed_subscriber = self.create_subscription (
            Float64MultiArray,
            '/correctPath',
            self.calculateLinearVelocities,
            1
        )

        self.derivatives_subscriber = self.create_subscription (
            Float64MultiArray,
            '/derivatives',
            self.updateDerivatives,
            1
        )

        self.points_subscriber = self.create_subscription (
            Float64MultiArray,
            '/path',
            self.updatePoints,
            1
        )

        self.velocity_publisher = self.create_publisher (
            Float64MultiArray,
            '/velocity',
            1
        )

    def updatePoints(self, message): # This is the points outputted by A*
        pointList = message.data 

        finalPointList = []
        x = 0
        
        for i, value in enumerate(pointList):
            if i % 2 == 0:
                x = value
            else:
                finalPointList.append([(x + 0.5) * 0.03, (value + 0.5) * 0.03]) # The + 0.5 puts the waypoint in the center of its square, the same as the curve smoother.

        self.pointsList = finalPointList # Outputs a full list of all the points [p0, p1, p2, p3, ...]

        if self.pendingMessage is not None:
            self.calculateLinearVelocities(self.pendingMessage)

    def updateDerivatives(self, message): # This is the derivatives outputted by the path planning algorithm
        derivatives = message.data
        derivativesList = []

        M1 = 0

        for i, value in enumerate(derivatives):
            if i % 2 == 0:
                M1 = value
            else:
                derivativesList.append([M1, value])

        self.derivativesList = derivativesList # Every row is one tangent [x, y]. Segment s uses row 2 * s as its first tangent and row 2 * s + 1 as its second.

        if self.pendingMessage is not None:
            self.calculateLinearVelocities(self.pendingMessage)

    def constructPointList(self, pointsArray): # This is the list of points outputted along the spline
        organizedPointList = []
        x = 0
        
        for i, value in enumerate(pointsArray):
            if i % 2 == 0:
                x = value
            else:
                organizedPointList.append([x, value])

        return organizedPointList

    def calculateFirstDerivative(self, u, v1, m1, v2, m2): # v1, m1, v2, m2 are all [x, y] arrays, and this returns [x', y'].
        return ((6 * u ** 2) - 6 * u) * v1 + (( 3 * u ** 2) - 4 * u + 1) * m1 + ((-6 * u ** 2) + 6 * u) * v2 + ((3 * u ** 2) - 2 * u) * m2

    def calculateSecondDerivative(self, u, v1, m1, v2, m2):
        return (12 * u - 6) * v1 + (6 * u  - 4) * m1 + (-12 * u + 6) * v2 + (6 * u - 2) * m2

    def calculateLinearVelocities(self, message):
        points = np.array(message.data)
        splinePointList = self.constructPointList(points)

        segmentCount = (len(splinePointList) - 1) // 100

        # The spline has the first point plus 100 points per segment, and every segment needs its waypoints and 2 tangents.
        if len(splinePointList) < 2 or (len(splinePointList) - 1) % 100 != 0 or len(self.pointsList) != segmentCount + 1 or len(self.derivativesList) != 2 * segmentCount:
            print("WAITING FOR THE MATCHING WAYPOINTS AND DERIVATIVES")
            self.pendingMessage = message
            return

        self.pendingMessage = None
    
        velocityList = [self.STARTVELOCITY] # We do not need a real velocity for the starting position, since that is the position underneath the car anyways.
        distanceList = [0.0]

        for i, point in enumerate(splinePointList):
            if i == 0:
                continue

            startPointIndex = (i - 1) // 100 # Which segment this point is on. This is found first so the first point of a new segment uses its own segment.
            k = (i - 1) % 100 + 1
            u = k / 100

            previousDistance = distanceList[i - 1]
            prevoiusPoint = splinePointList[i - 1]  

            newDistance = previousDistance + np.sqrt((point[0] - prevoiusPoint[0]) ** 2 + (point[1] - prevoiusPoint[1]) ** 2)

            p1 = np.array(self.pointsList[startPointIndex])
            p2 = np.array(self.pointsList[startPointIndex + 1])
            m1 = np.array(self.derivativesList[2 * startPointIndex])
            m2 = np.array(self.derivativesList[2 * startPointIndex + 1])

            firstDerivative = self.calculateFirstDerivative(u, p1, m1, p2, m2)
            secondDerivative = self.calculateSecondDerivative(u, p1, m1, p2, m2)

            speedCubed = max(np.linalg.norm(firstDerivative) ** 3, 1e-9)

            curvature = np.abs(firstDerivative[0] * secondDerivative[1] - firstDerivative[1] * secondDerivative[0]) / speedCubed

            if abs(curvature) < 1e-6:
                curveVel = self.MAXVELOCITY
            else:
                curveVel = min(self.MAXVELOCITY, np.sqrt(self.MAXACCELERATION / abs(curvature)), self.MAXVELOCITYANGULAR / abs(curvature))

            distanceList.append(newDistance)
            velocityList.append(max(self.MINVELOCITY, curveVel))

        for i, velocity in enumerate(velocityList):
            if i == 0:
                continue

            prevVel = velocityList[i - 1]
            deltaDistance = distanceList[i] - distanceList[i - 1]

            velocityList[i] = min(velocity, np.sqrt((prevVel) ** 2 + 2 * self.MAXACCELERATION * deltaDistance))

        velocityList[-1] = 0.0 # The car has to stop at the end of the path. The backwards pass below spreads this out so the car brakes in time.

        for i in range(len(velocityList) - 2, -1, -1):
            if i == 0:
                continue

            deltaDistance = distanceList[i + 1] - distanceList[i]

            maxVChange = np.sqrt(velocityList[i + 1] ** 2 + 2 * self.MAXBREAK * deltaDistance)

            velocityList[i] = min(velocityList[i], maxVChange)

        print("WE POSTING VEOCITIES")

        velocityMsg = Float64MultiArray()
        velocityMsg.data = [float(velocity) for velocity in velocityList]
        self.velocity_publisher.publish(velocityMsg)

def main(args = None):
    print("WE CALCULATING LINEAR VELOCITIES")

    rclpy.init(args = args)

    linearVelocityNode = calculateLinearVelocities()

    rclpy.spin(linearVelocityNode)

    linearVelocityNode.destroy_node()

    rclpy.shutdown()

if __name__ == '__main__':
    main()