import rclpy
import numpy as np
import sys
from collections import deque

from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Float64

class findFrontiers(Node):
    def __init__(self):
        super().__init__("findFrontiers")

        self.stateVector = np.zeros(7) 
        self.occupancyDict = {}
        
        self.stateSubscriber = self.create_subscription (
            Float64MultiArray,
            '/corrected',
            self.updateStateVector,
            1
        )

        self.occupancySubscriber = self.create_subscription (
            Float64MultiArray,
            '/occupancyGrid',
            self.locateFrontiers,
            1
        )

    def updateStateVector(self, message):
        self.stateVector = np.array(message.data)

    def locateFrontiers(self, message):
        directions = [(-1, 1), (0, 1), (1, 1),
                      (-1, 0),         (1, 0),
                      (-1, -1), (0, -1), (1, -1)]

        seenLocateSet = set()
        frontierSet = set()
        locateQueue = deque()

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
                self.occupancyDict[(x, y)] = value

        startingSquare = (self.stateVector[0] // .03, self.stateVector[1] // .03)
        seenLocateSet.add(startingSquare)

        for d in directions:
            newSquare = (startingSquare[0] + d[0], startingSquare[1] + d[1])

            seenLocateSet.add(newSquare)
            locateQueue.append(newSquare)

        while locateQueue:
            newSquare = locateQueue.popleft()

            if self.occupancyDict.get(newSquare, -5) == -5: # if it is a frontier, do not continue searching its boundaries, just add it to frontier set.
                frontierSet.add(newSquare) # -5 is also outside of the possible vlaues for occupancyDict, since I clamp it at around -4.
                continue 

            if self.occupancyDict[newSquare] > 0: # if we have seen it before, skip searching; if it is a boarder, skip searching
                continue

            for d in directions: # append all the new squares surrounding it if it is not a frontier and is not a boarder
                updatedSquare = (newSquare[0] + d[0], newSquare[1] + d[1])
                
                if not (updatedSquare in seenLocateSet):
                    seenLocateSet.add(updatedSquare)
                    locateQueue.append(updatedSquare)

        
