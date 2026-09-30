import rclpy
import numpy as np
import math

from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64MultiArray
from slam_messages.msg import VoxelGrid
from slam_messages.msg import Voxel

class scanMatching(Node):
    def __init__(self):
        self.stateVector = None
        self.voxelDict = {}
        self.voxelSize = 20

        """stateVector:
            np.array([0,  x position
            0,  y position
            0,  rotation
            0,  linear velocity
            0,  angular velocity
            0,  linear bias
            0])  angular bias"""
        
        self.EKF_Subscriber = self.create_subscription (
            Float64MultiArray,
            '/prediction',
            self.updateState,
            1
        )

        self.scan_Subscriber = self.create_subscription (
            LaserScan,
            '/scan',
            self.createPosEstimate,
            1
        )

        self.voxel_Subscriber = self.create_subscription (
            VoxelGrid,
            '/NDTVoxelMap',
            self.updateVoxel,
            1
        )

    def updateState(self, message):
        self.stateVector = np.array(message.data)

    def updateVoxel(self, message):
        for voxel in message:
            self.voxelDict[(voxel.voxel_x, voxel.voxel_y)] = ([voxel.x_mean, voxel.y_mean], [[voxel.x_var, voxel.xy_cov], [voxel.xy_cov, voxel.y_var]])

    def calculateJ(self, pX, pY, theta):
        return [[ 1, 0, -pX * np.sin(theta) - pY * np.cos(theta)],
                [0, 1, pX * np.cos(theta) - pY * np.sin(theta)]]

    def calculateH(self, pX, pY, theta):
        return [[-pX * np.cos(theta) + pY * np.sin(theta)],
                [-pX * np.sin(theta) - pY * np.cos(theta)]]

    def calculateS(self, J, covariance, point, mean):
        return J.T @ covariance @ (point - mean)

    def calculateK(self, H, covariance, point, mean):
        return (point - mean).T @ covariance @ H

    def calculateQ(self, J, covariance, K):
        Q = J.T @ covariance @ J
        Q[2, 2] += K

        return Q

    def createPosEstimate(self, message):
        if self.voxelDict != {}:
            deltaPos = np.array([100, 100, 100])
            iterationCount = 0

            while np.linalg.norm(deltaPos) > 0.001 and iterationCount < 100:
                robotPosVect = [self.stateVector[0], self.stateVector[1]]
                totalError = 0
                gradient = np.zeros(3)
                Hessian =  np.zeros((3, 3))

                for i, lidarRange in enumerate(message.ranges):
                    if lidarRange > 20:
                        continue
                
                    angle = message.angle_min + i * message.angle_increment # This is the number, in radians, my lidar says it angle increases by each increment.

                    changePosVect = [lidarRange * np.cos(angle), lidarRange * np.sin(angle)]

                    truePosVect = [[np.cos(self.stateVector[2]), -np.sin(self.stateVector[2])], [np.sin(self.stateVector[2]), np.cos(self.stateVector[2])]] @ changePosVect + robotPosVect # True position represented as a vector.

                    voxelX = truePosVect[0] // self.voxelSize
                    voxelY = truePosVect[1] // self.voxelSize

                    voxel = self.voxelDict.get((voxelX, voxelY))

                    if voxel is None:
                        continue
                    
                    A = np.linalg.inv([[voxel.x_var, voxel.xy_cov],
                                         [voxel.xy_cov, voxel.y_var]])

                    mean = [voxel.x_mean, voxel.y_mean]

                    posDiffVect = truePosVect - mean

                    distance = (posDiffVect).T @ A @ (posDiffVect)

                    error = math.exp(-distance / 2)
                    totalError += error

                    J = self.calculateJ(changePosVect[0], changePosVect[1], self.stateVector[2])
                    h = self.calculate_h(changePosVect[0], changePosVect[1], self.stateVector[2])
                    S = self.calculateS(J, A, truePosVect, mean)
                    K = self.calculateK(h, A, truePosVect, mean)
                    Q = self.calculateQ(J, A, K)

                    gradient += error * S
                    Hessian += error * (Q - np.outer(S, S))

                deltaPos = -np.linalg.solve(Hessian, gradient)
                updatedVector = self.stateVector + deltaPos
                
                print("UPDATED VECTOR: ", updatedVector)
                print("PREVIOUS VECTOR: ", self.stateVector)
                
                self.stateVector = updatedVector
                iterationCount += 1

                




