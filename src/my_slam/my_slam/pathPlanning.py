import rclpy
import numpy as np
import heapq

from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Point
from std_msgs.msg import Bool

class pathPlanning(Node):
    def __init__(self):
        super().__init__("pathPlanning")

        self.stateVector = np.zeros(7)
        self.inflatedOccupancyDict = {}
        self.shortestDict = {}
        self.clearanceCache = {}

        self.haveState = False
        self.haveTarget = False

        self.targetSquare = (0, 0)
        self.startSquare = (0, 0)
        self.STARTCLEARANCE = 4 # Squares around the robot that count as free, since the robot can start inside the inflation. Keep this the same as GRACESQUARES in the path checker.
        self.CLEARANCEWEIGHT = 1.3 # Extra cost for every blocked square touching a square. This pushes the path away from the inflated edge.
        self.MINCLEARANCE = 2 # A square can only be used if no known occupied square is within this many squares of it. gaps smaller than (2 * MINCLEARANCE + 1) squares become unpassable. 0 turns the rule off.
        self.TARGETGRACE = 2 # Squares around the target where the MINCLEARANCE rule is skipped, since the target sits at the edge of the known map.

        self.updatePath = True

        self.directions = [(-1, 1, 1.41421), (0, 1, 1), (1, 1, 1.41421),
                            (-1, 0, 1),                 (1, 0, 1),
                            (-1, -1, 1.41421), (0, -1, 1), (1, -1, 1.41421)]

        self.state_subscriber = self.create_subscription (
            Float64MultiArray,
            '/corrected',
            self.updateState,
            1
        )

        self.inflated_occupancy_subscriber = self.create_subscription (
            Float64MultiArray,
            '/inflatedOccupancyGrid',
            self.updateOccupancyGrid,
            1
        )

        self.frontier_subscriber = self.create_subscription (
            Point,
            '/frontier',
            self.updateTarget,
            1
        )

        self.path_update_subscriber = self.create_subscription (
            Bool,
            '/updatePath',
            self.needNewPath,
            1
        )

        self.path_publisher = self.create_publisher (
            Float64MultiArray,
            '/path',
            1
        )

        self.unfiltered_publisher = self.create_publisher (
            Float64MultiArray,
            '/unfilteredPath',
            1
        )

        self.pathTimer = self.create_timer(
            0.1,
            self.checkPathRequest
        )

        self.invalid_frontier_publisher = self.create_publisher (
            Point,
            '/invalidFrontier',
            1
        )

    def updateTarget(self, message):
        self.targetSquare = (int(message.x), int(message.y)) # This is in the form of a tuple of whole squares.
        self.haveTarget = True

    def checkPathRequest(self):
        if self.updatePath:
            if not self.haveState:
                return

            if not self.haveTarget:
                return

            if not self.inflatedOccupancyDict:
                return
            
            self.updatePath = False

            self.createPath()

    def needNewPath(self, message):
        self.updatePath = True

    def updateState(self, message):
        self.stateVector = np.array(message.data)
        self.haveState = True

    def updateOccupancyGrid(self, message):
        grid = np.array(message.data).reshape(-1, 3) # Every row is [x, y, logOdds]

        squares = zip(grid[:, 0].astype(int).tolist(), grid[:, 1].astype(int).tolist())
        self.inflatedOccupancyDict = dict(zip(squares, grid[:, 2].tolist())) # A brand new dictionary every time, so squares that are no longer occupied disappear.

    def isBlocked(self, x, y):
        if max(abs(x - self.startSquare[0]), abs(y - self.startSquare[1])) <= self.STARTCLEARANCE:
            return False # The robot is sitting here, so these squares cant be treated as blocked.

        return self.inflatedOccupancyDict.get((x, y), 1) > 0 # Default value is 1 so that any square not inside the occupancy dict counts as blocked.

    def nearBlocked(self, x, y): # Counts how many of the 8 squares around (x, y) are blocked.
        count = 0

        for d in self.directions:
            if self.isBlocked(x + d[0], y + d[1]):
                count += 1

        return count

    def isKnownOccupied(self, x, y): # Same as isBlocked, except a square that is not inside the occupancy dict counts as unknown instead of blocked.
        if max(abs(x - self.startSquare[0]), abs(y - self.startSquare[1])) <= self.STARTCLEARANCE:
            return False # start exemption

        return self.inflatedOccupancyDict.get((x, y), -1) > 0 # Default value is -1 so that unknown squares are NOT treated as occupied. Otherwise every square next to unexplored space would fail the clearance check.

    def hasClearance(self, x, y): # Checks that no known occupied square is within MINCLEARANCE squares of (x, y)
        if max(abs(x - self.targetSquare[0]), abs(y - self.targetSquare[1])) <= self.TARGETGRACE:
            return True

        if (x, y) in self.clearanceCache:
            return self.clearanceCache[(x, y)]

        result = True

        for dx in range(-self.MINCLEARANCE, self.MINCLEARANCE + 1):
            for dy in range(-self.MINCLEARANCE, self.MINCLEARANCE + 1):
                if self.isKnownOccupied(x + dx, y + dy):
                    result = False
                    break

            if not result:
                break

        self.clearanceCache[(x, y)] = result # Every square only gets scanned once per plan.

        return result

    def calculateDistance(self, start, target):
        dx = abs(target[0] - start[2]) # The square has score stored in position 0.
        dy = abs(target[1] - start[3])

        return (dx + dy) + (np.sqrt(2) - 2) * min(dx, dy)

    def determineMoveOrder(self, targetSquare):
        currentSquare = targetSquare
        finalMovesList = []

        iterationCount = 0
        
        while True:
            iterationCount += 1
            
            finalMovesList.append((currentSquare[0], currentSquare[1]))
            previousSquare = self.shortestDict[currentSquare] # ShortestDict takes coordinate pairs as keys, so I have to exclude score when updating currentSquare.

            if previousSquare == None:
                finalMovesList.reverse()

                message = Float64MultiArray()
                message.data = np.array(finalMovesList, dtype=float).flatten().tolist()
                self.unfiltered_publisher.publish(message) # This is published BEFORE /path so the path checker already has every square by the time the first smoothed path comes back.

                self.prepareMoveList(finalMovesList)

                break

            currentSquare = (previousSquare[0], previousSquare[1])


    def prepareMoveList(self, moveList):
        finalList = []

        if len(moveList) > 3:
            previousChange = None
            previousMove = moveList[0]

            for move in moveList[1:]:
                currentChange = (move[0] - previousMove[0], move[1] - previousMove[1])
                if currentChange != previousChange:
                    finalList.append(previousMove)

                previousMove = move
                previousChange = currentChange

            finalList.append(moveList[-1])
        else:
            finalList = moveList
        
        msg = Float64MultiArray()
        msg.data = np.array(finalList, dtype=float).flatten().tolist()
        self.path_publisher.publish(msg)
    
    def createPath(self):
        self.shortestDict.clear()
        self.clearanceCache.clear() # The grid changes between plans, so old clearance answers can't be reused.

        startX = int(np.floor(self.stateVector[0] / .03))
        startY = int(np.floor(self.stateVector[1] / .03))
        self.startSquare = (startX, startY)

        if self.targetSquare == self.startSquare:
            return

        if self.isBlocked(self.targetSquare[0], self.targetSquare[1]):
            return

        robotSquare = [0, 0.0, startX, startY] # squares in the queue are saved as [score, distanceTraveled, x, y]. The parent of every square lives in shortestDict.
        robotSquare[0] = self.calculateDistance(robotSquare, self.targetSquare) * 1.2 # Initializes the score of robotSquare.

        gScore = {self.startSquare: 0.0} # Cheapest known distanceTraveled for every square we have reached.
        self.shortestDict[self.startSquare] = None

        squareHeap = []
        heapq.heapify(squareHeap)
        heapq.heappush(squareHeap, robotSquare)

        while squareHeap:
            startSquare = heapq.heappop(squareHeap)
            currentSquare = (startSquare[2], startSquare[3])

            if startSquare[1] > gScore[currentSquare]: # A cheaper way to this square was found after this entry was pushed, so this entry is outdated.
                continue

            if currentSquare == self.targetSquare: # The goal is only checked when it is popped, which is what makes the path the cheapest one.
                self.determineMoveOrder(self.targetSquare)
                return

            for d in self.directions:
                # [dx, dy, dTraveled] = d

                newX = startSquare[2] + d[0]
                newY = startSquare[3] + d[1]

                if self.isBlocked(newX, newY):
                    continue

                if not self.hasClearance(newX, newY): # A square with a known occupied square right next to it is part of a gap that is too narrow, so A* can't use it. If this is the only way to the target, the loop ends and the target is reported as invalid below.
                    continue

                if d[0] != 0 and d[1] != 0: # A diagonal move can't squeeze between two blocked squares, so both squares beside it have to be free.
                    if self.isBlocked(startSquare[2] + d[0], startSquare[3]) or self.isBlocked(startSquare[2], startSquare[3] + d[1]):
                        continue

                newT = startSquare[1] + d[2]

                if newT >= gScore.get((newX, newY), np.inf): # Cheap check first, because adding the clearance cost can only make newT bigger.
                    continue

                newT += self.CLEARANCEWEIGHT * self.nearBlocked(newX, newY)

                if newT >= gScore.get((newX, newY), np.inf):
                    continue

                distance = self.calculateDistance((0, 0, newX, newY), self.targetSquare)

                newScore = newT + distance * 1.2

                gScore[(newX, newY)] = newT
                self.shortestDict[(newX, newY)] = (startSquare[2], startSquare[3], newT)
                heapq.heappush(squareHeap, [newScore, newT, newX, newY])

        msg = Point()
        msg.x = float(self.targetSquare[0])
        msg.y = float(self.targetSquare[1])
        self.invalid_frontier_publisher.publish(msg)

def main(args = None):
    print("WE PATH PLANNING")

    rclpy.init(args=args)

    pathNode = pathPlanning()

    rclpy.spin(pathNode)

    pathNode.destroy_node()

    rclpy.shutdown()

if __name__ == '__main__':
    main()