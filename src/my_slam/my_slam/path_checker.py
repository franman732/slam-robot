import rclpy
import numpy as np

from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

class pathChecker(Node):
    def __init__(self):
        super().__init__("pathChecker")

        self.unfilteredPath = [] # Every square A* walked through, as a list of (x, y) whole squares.
        self.cellIndex = {} # square --> its position inside unfilteredPath
        self.filteredPath = [] # The current list of waypoints, as (x, y) whole squares.
        self.inflatedOccupancyDict = {}
        self.pendingMessage = None # A smoothed path that showed up before the waypoints that made it. It gets checked as soon as they arrive.
        self.passes = 0

        self.SQUARESIZE = 0.03
        self.GRACESQUARES = 4 # Spline points this close to the start are never flagged, since the robot can start inside the inflation. Keep this the same as STARTCLEARANCE in the path planner.
        self.MAXPASSES = 40

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

    def toCells(self, data): # Turns a flat [x0, y0, x1, y1, ...] list into a list of (x, y) whole squares.
        pairs = np.array(data).reshape(-1, 2)

        return [(int(round(x)), int(round(y))) for x, y in pairs]

    def updateInflatedOccupancy(self, message): # Updates the occupancy grid, which is a dictionary of (x, y) --> log value. log value > 0.5 means occupied. x and y are in square coordinates.
        grid = np.array(message.data).reshape(-1, 3) # Every row is [x, y, logOdds]

        squares = zip(grid[:, 0].astype(int).tolist(), grid[:, 1].astype(int).tolist())
        self.inflatedOccupancyDict = dict(zip(squares, grid[:, 2].tolist())) # A brand new dictionary every time, so squares that are no longer inflated disappear.

    def updateUnfilteredPath(self, message): # Updates the unfiltered path outputted by A*. This is the path that includes all the extra points between waypoints in the form of a list of points.
        self.unfilteredPath = self.toCells(message.data) # All the values are in square coordinates.
        self.cellIndex = {square: i for i, square in enumerate(self.unfilteredPath)}
        self.passes = 0 # A new A* path means a fresh start for the repair passes.

    def updatePath(self, message): # This function updates the filtered path outputted by A* that only includes the waypoints
        self.filteredPath = self.toCells(message.data)

        if self.pendingMessage is not None:
            self.checkPath(self.pendingMessage)

    def newWaypoint(self, lowerWaypoint, upperWaypoint): # This function takes the two waypoints containing the error point, and produces a new point to fill it.
        lowerIndex = self.cellIndex.get(lowerWaypoint)
        upperIndex = self.cellIndex.get(upperWaypoint)

        if lowerIndex is None or upperIndex is None or upperIndex - lowerIndex < 2:
            return None # There is no A* square left in between these two waypoints to add.

        return self.unfilteredPath[(lowerIndex + upperIndex) // 2]

    def checkPath(self, msg): # This function updates the path filled in by the spline curve smoothing. It creates a list of all the waypoints with 99 additional points in between.
        print("WE CHECKING PATH")  # This function has coordinates in global position. 
        pathPoints = np.array(msg.data).reshape(-1, 2)
        waypointCount = len(self.filteredPath)

        # The smoother outputs the first waypoint plus 100 points for every segment, so this only checks a smoothed path that was made from the waypoints we currently have.
        if waypointCount < 2 or len(pathPoints) != 1 + 100 * (waypointCount - 1):
            print("SMOOTHED PATH DOES NOT MATCH THE WAYPOINTS YET, WAITING")
            self.pendingMessage = msg
            return

        self.pendingMessage = None

        if not np.isfinite(pathPoints).all():
            print("NAN IN THE SPLINE, THERE ARE PROBABLY DUPLICATE WAYPOINTS")
            return

        squares = np.floor(pathPoints / self.SQUARESIZE).astype(int)
        startX = int(squares[0][0])
        startY = int(squares[0][1])

        badSegments = set()

        for k, square in enumerate(squares):
            squareX = int(square[0])
            squareY = int(square[1])

            if max(abs(squareX - startX), abs(squareY - startY)) <= self.GRACESQUARES:
                continue

            if self.inflatedOccupancyDict.get((squareX, squareY), 1) > 0: # Same default as A*, so a square A* thinks is free is also free here.
                badSegments.add(max(0, (k - 1) // 100)) # Point 0 is the start, then every segment adds 100 points.

        if len(badSegments) == 0:
            print("FINAL PATH FOUND")
            self.finalPathPublisher.publish(msg)
            return

        updatedPath = False

        for segment in sorted(badSegments, reverse = True): # Going backwards so inserting a waypoint never shifts the segments we still have to fix.
            newPoint = self.newWaypoint(self.filteredPath[segment], self.filteredPath[segment + 1])

            if newPoint is None:
                continue

            self.filteredPath.insert(segment + 1, newPoint) # filtered path is the path outputted by A* that has only waypoints.
            updatedPath = True

        self.passes += 1

        if not updatedPath or self.passes > self.MAXPASSES:
            print("CAN'T REPAIR SEGMENTS: ", sorted(badSegments), " REPLAN WITH MORE CLEARANCE")
            return

        print("PUBLISHING CORRECTED PATH")
        correctedPath = Float64MultiArray()
        correctedPath.data = [float(value) for square in self.filteredPath for value in square]
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