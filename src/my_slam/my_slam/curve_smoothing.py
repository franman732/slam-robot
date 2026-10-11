import rclpy
import numpy as np

from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

class curveSmoothing(Node):
    def __init__(self):
        super().__init__("curveSmoothing")

        self.path_subcriber = self.create_subscription (
            Float64MultiArray,
            '/path',
            self.smoothPath,
            1
        )

        self.smoothed_publisher = self.create_publisher (
            Float64MultiArray,
            '/smoothedPath',
            1
        )

        self.derivatives_publisher = self.create_publisher (
            Float64MultiArray,
            '/derivatives',
            1
        )

    def getKnotPoints(self, prevKnot, p1, p0):
        return prevKnot + np.linalg.norm(p1 - p0) ** 0.5

    def calculateCurveValue(self, u, p1, m1, p2, m2):
        return p1 * ((2 * u ** 3) - (3 * u ** 2) + 1) + m1 * (u ** 3 - (2 * u ** 2) + u) + p2 * ((-2 * u ** 3) + (3 * u ** 2)) + m2 * (u ** 3 - u ** 2)
    
    def getCatmullRomPositions(self, p0, p1, p2, p3):
        pointList = []
        
        t0 = 0
        t1 = self.getKnotPoints(t0, p1, p0)
        t2 = self.getKnotPoints(t1, p2, p1)
        t3 = self.getKnotPoints(t2, p3, p2)

        d1 = (p1 - p0) / (t1 - t0) - (p2 - p0) / (t2 - t0) + (p2 - p1) / (t2 - t1)
        d2 = (p2 - p1) / (t2 - t1) - (p3 - p1) / (t3 - t1) + (p3 - p2) / (t3 - t2)

        deltaT = t2 - t1

        m1 = deltaT * d1
        m2 = deltaT * d2

        for i in range(1, 101): # gives me 100 points, including the ending point.
            u = i / 100

            pointList.append(self.calculateCurveValue(u, p1, m1, p2, m2))

        return pointList, [m1, m2]

    def smoothPath(self, message):
        wayPoints = []
        smoothedPointsList = []
        derivativesList = []
        pInitial = None
        pFinal = None
        x = 0

        for i, value in enumerate(message.data):
            if i % 2 == 0:
                x = value * 0.03
            else:
                wayPoints.append(np.array([x, value * 0.03]))

        for i in range(2):
            if i % 2 == 0:
                x = 2 * wayPoints[0][0] - wayPoints[1][0] # Extrapolates a starting point for curve smoothing
            else:
                pInitial = np.array([x, 2 * wayPoints[0][1] - wayPoints[1][1]])

        for i in range(2):
            if i % 2 == 0:
                x = 2 * wayPoints[len(wayPoints) - 1][0] - wayPoints[len(wayPoints) - 2][0] # Extrapolates an extra point at the end for curve smoothing
            else: 
                pFinal = np.array([x, 2 * wayPoints[-1][1] - wayPoints[-2][1]])
        
        wayPoints.insert(0, pInitial) 
        wayPoints.append(pFinal)

        smoothedPointsList.append(wayPoints[1].copy())

        for i in range(1, len(wayPoints) - 2):
            p0 = wayPoints[i - 1]
            p1 = wayPoints[i]
            p2 = wayPoints[i + 1]
            p3 = wayPoints[i + 2]

            pointList, derivatives = self.getCatmullRomPositions(p0, p1, p2, p3)

            smoothedPointsList.extend(pointList)
            derivativesList.extend(derivatives)

        print("WE PRINTING")

        derivativesMsg = Float64MultiArray()
        derivativesMsg.data = np.array(derivativesList).flatten().tolist()
        self.derivatives_publisher.publish(derivativesMsg)

        pointsMsg = Float64MultiArray()
        pointsMsg.data = np.array(smoothedPointsList).flatten().tolist()
        self.smoothed_publisher.publish(pointsMsg)

def main(args = None):
    print("WE CURVE SMOOTHING")

    rclpy.init(args = args)

    curveNode = curveSmoothing()

    rclpy.spin(curveNode)

    curveNode.destroy_node()

    rclpy.shutdown()

if __name__ == '__main__':
    main()