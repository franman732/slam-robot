import rclpy
import numpy as np

from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

class pathChecker(Node):
    def __init__(self):
        super().__init__("pathChecker")

        self.unfilteredPath = []
        self.filteredPath = []
        self.inflatedOccupancyDict = {}

        self.unfilteredSubscriber = self.create_subscription (
            Float64MultiArray,
            '/unfilteredPath',
            self.updateUnfilteredPath,
            1
        )

        self.filteredSubscriber = self.create_subscription (
            Float64MultiArray,
            '/path',
            self.updatePath,
            1
        )

        self.inflatedOccupancySubscriber = self.create_subscription (
            Float64MultiArray,
            '/inflatedOccupancyGrid',
            self.updateInflatedOccupancy,
            1
        ) 

        self.smoothedSubscriber = self.create_subscription (
            Float64MultiArray,
            '/smoothedPath',
            self.checkPath,
            1
        )

        self.correctedPathPublisher = self.create_publisher (
            Float64MultiArray,
            '/path',
            1
        )

        self.finalPathPublisher = self.create_publisher (
            Float64MultiArray,
            '/correctPath',
            1
        )

    def updateInflatedOccupancy(self, message): # Updates the occupancy grid, which is a dictionary of (x, y) --> log value. log value > 0.5 means occupied. x and y are in square coordinates.
        gridList = message.data
        x = 0
        y = 0

        for i, value in enumerate(gridList):
            identifier = (i + 1) % 3

            if identifier == 1:
                x = value
            elif identifier == 2:
                y = value
            else:
                self.inflatedOccupancyDict[(x, y)] = value

    def updateUnfilteredPath(self, message): # Updates the unfiltered path outputted by A*. This is the path that includes all the extra points between waypoints in the form of a list of points.
        unfiltered = message.data            # All the values are in square coordinates.
        finalUnfiltered = []

        x = 0

        for i, value in enumerate(unfiltered):
            if i % 2 == 0:
                x = value
            else:
                finalUnfiltered.append([x, value])

        self.unfilteredPath = finalUnfiltered

    def updatePath(self, message): # This function updates the filtered path outputted by A* that only includes the waypoints
        pathMessage = message.data 
        finalPath = []

        x = 0

        for i, value in enumerate(pathMessage):
            if i % 2 == 0:
                x = value
            else:
                finalPath.append([x, value])

        self.filteredPath = finalPath

    def newWaypoint(self, lowerWaypoint, upperWaypoint): # This function takes the two waypoints containing the error point, and produces a new point to fill it.
        foundLower = False                               
        midIndex = 0
        lowerIndex = 0

        for i, point in enumerate(self.unfilteredPath):
            if point == lowerWaypoint:
                foundLower = True
                lowerIndex = i
                midIndex = lowerIndex

            if foundLower:
                if (i - lowerIndex) % 2 == 0: # Every two steps forward, increase our mid position by 1.
                    midIndex += 1

            if point == upperWaypoint:
                return self.unfilteredPath[midIndex]
                    


    def checkPath(self, msg): # This function updates the path filled in by the spline curve smoothing. It creates a list of all the waypoints with 99 additional points in between.
        path = msg.data       # This function has coordinates in global position. 
        updatedPath = False
        
        for i, value in enumerate(path):
            if i % 2 == 0:
                x = np.floor(value // .03 + 1e-6)
            else:
                square = (x, np.floor(value // .03 + 1e-6))

                if self.inflatedOccupancyDict.get(square, 1) > 0:
                    lowerWaypoint = path[i // 100]
                    upperWaypoint = path[lowerWaypoint + 1]

                    newPoint = self.newWaypoint(lowerWaypoint, upperWaypoint)

                    self.filteredPath.insert(upperWaypoint, newPoint) # filtered path is the path outputted by A* that has only waypoints.
                    updatedPath = True

        if updatedPath == False:
            self.finalPathPublisher.publish(msg)
        else:
            correctedPath = Float64MultiArray()
            correctedPath.data = np.array(self.filteredPath).flatten().tolist()
            self.correctedPathPublisher.publish(correctedPath)

def main(args = None):
    print("WE CHECKING THE PATH")

    rclpy.init(args = args)

    pathCheckNode = pathChecker()

    rclpy.spin(pathCheckNode)

    pathCheckNode.destroy_node()

    rclpy.shutdown()

if __name__ == '__main__':
    main()

    


        

