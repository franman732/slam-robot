import rclpy
import numpy as np
import math

from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float64MultiArray
from slam_messages.msg import VoxelGrid
from std_msgs.msg import Float64
from std_msgs.msg import Bool

class scanMatching(Node):
    def __init__(self):
        super().__init__('map_matching_node') 

        self.stateVector = None
        self.updatedVector = None
        self.voxelDict = {}
        self.voxelSize = .20
        self.lambdaRegress = 1e-4
        self.maxTranslation = 0.05
        self.maxRotation = 0.05
        self.lidarOffset = np.array([-0.032, 0.0])

        self.minEigenvalue = 1e-3
        self.degenerateRatio = 0.05
        self.maxTotalTranslation = 0.15

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

        self.correction_scan_publisher = self.create_publisher (
            Bool,
            'correctionScan',
            1
        )

        self.correction_publisher = self.create_publisher (
            Float64MultiArray,
            '/correction',
            1
        )

        self.error_publisher = self.create_publisher (
            Float64,
            '/NDT_error',
            1
        )

    def updateState(self, message):
        self.stateVector = np.array(message.data)

    def updateVoxel(self, message):
        for voxel in message.voxels:
            self.voxelDict[(voxel.voxel_x, voxel.voxel_y)] = (np.array([voxel.x_mean, voxel.y_mean]), np.array([[voxel.x_var, voxel.xy_cov], [voxel.xy_cov, voxel.y_var]]))

    def calculateJ(self, pX, pY, theta):
        return np.array([[ 1, 0, -pX * np.sin(theta) - pY * np.cos(theta)],
                        [0, 1, pX * np.cos(theta) - pY * np.sin(theta)]])

    def calculateH(self, pX, pY, theta):
        return np.array([[-pX * np.cos(theta) + pY * np.sin(theta)],
                        [-pX * np.sin(theta) - pY * np.cos(theta)]])

    def calculateS(self, J, covariance, point, mean):
        return J.T @ covariance @ (point - mean)

    def calculateK(self, H, covariance, point, mean):
        return (point - mean).T @ covariance @ H

    def calculateQ(self, J, covariance, K):
        Q = J.T @ covariance @ J
        Q[2, 2] += K

        return Q

    def wrapAngle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def createPosEstimate(self, message):
        if self.voxelDict != {} and self.stateVector is not None:
            deltaPos = np.array([0, 0, 0])
            iterationCount = 0
            storageList = []
            keepUpdating = True
            validUpdate = True

            startVector = self.stateVector.copy() # top of the if-block, before the while loop
            
            correctionMsg = Bool()  
            correctionMsg.data = True # This just publishes any message to tell all the map building nodes that the scan published when this message is received is the one that matches the outputted corrected state.
            self.correction_scan_publisher.publish(correctionMsg) 

            while keepUpdating and iterationCount < 100:
                robotPosVect = np.array([self.stateVector[0], self.stateVector[1]])
                totalError = 0
                gradient = np.zeros(3)
                Hessian =  np.zeros((3, 3))
                totalNodes = 0
                matchedNodes = 0 # next to totalNodes = 0 inside the while loop

                for i, lidarRange in enumerate(message.ranges):
                    if lidarRange > 20 or math.isnan(lidarRange):
                        continue
                
                    angle = message.angle_min + i * message.angle_increment # This is the number, in radians, my lidar says it angle increases by each increment.

                    changePosVect = np.array([lidarRange * np.cos(angle), lidarRange * np.sin(angle)])
                    #print("CHANGEPOSVECT: ", changePosVect)

                    basePoint = changePosVect + self.lidarOffset

                    truePosVect = [[np.cos(self.stateVector[2]), -np.sin(self.stateVector[2])], [np.sin(self.stateVector[2]), np.cos(self.stateVector[2])]] @ basePoint + robotPosVect # True position represented as a vector.

                    #print("TRUE POS VECT: ", truePosVect)

                    voxelX = truePosVect[0] // self.voxelSize
                    voxelY = truePosVect[1] // self.voxelSize

                    voxel = self.voxelDict.get((voxelX, voxelY), None)

                    totalNodes += 1

                    if voxel is None:
                        continue

                    matchedNodes += 1

                    #print("WE PASSED VOXEL ERROR")
                    #print("VOXEL: ", voxel)

                    covarianceMatrix = voxel[1]
                    covarianceMatrixReg = covarianceMatrix + np.eye(2) * 1e-4
                    
                    A = np.linalg.inv(covarianceMatrixReg)

                    mean = voxel[0]

                    posDiffVect = truePosVect - mean

                    distance = (posDiffVect).T @ A @ (posDiffVect)

                    error = math.exp(-distance / 2)
                    totalError += error

                    #print("TOTAL ERROR: ---------------", totalError)

                    J = self.calculateJ(basePoint[0], basePoint[1], self.stateVector[2])
                    h = self.calculateH(basePoint[0], basePoint[1], self.stateVector[2])
                    S = self.calculateS(J, A, truePosVect, mean)
                    K = self.calculateK(h, A, truePosVect, mean)
                    Q = self.calculateQ(J, A, K)

                    gradient += error * S
                    Hessian += error * (Q - np.outer(S, S))

                #print("PRE UPDATE HESSIAN: ", Hessian)
                #print("PRE UPDATE GRADIENT: ", gradient)
                #print("NUM OF NODES: ", totalNodes)
                #print("------------------------------------------------------------")

                if matchedNodes < 50:
                    print("too few points")
                    validUpdate = False
                    break

                gradientNorm = gradient / totalNodes
                hessianNorm = Hessian / totalNodes  

                Hessian_reg = hessianNorm + self.lambdaRegress * np.eye(3)

                if not np.all(np.isfinite(Hessian_reg)): # Moved above the eigen decomposition, since that can throw on NaN.
                    print("invalid Hessian")
                    validUpdate = False
                    break

                if not np.all(np.isfinite(gradientNorm)):
                    print("invalid gradient")
                    validUpdate = False
                    break

                eigenvalues, eigenvectors = np.linalg.eigh(Hessian_reg)

                # Not positive definite is normal while the guess is still off. abs() turns negative curvature into positive curvature so the step still goes downhill,
                # and the floor stops tiny eigenvalues from making huge steps. This replaces the positive definite check, the condition check, and the solve.
                stepEigenvalues = np.maximum(np.abs(eigenvalues), self.minEigenvalue)

                deltaPos = -eigenvectors @ ((eigenvectors.T @ gradientNorm) / stepEigenvalues)

                """print("RAW DELTA:", deltaPos)
                print("RAW TRANSLATION:", np.linalg.norm(deltaPos[:2]))

                print("------------------------------------------------------------")
                print("POST UPDATE HESSIAN: ", Hessian_reg)
                print("POST UPDATE GRADIENT: ", gradientNorm)
                print("-----------------------------------------------------------")"""

                translation_norm = np.linalg.norm(deltaPos[:2])
                rotation_norm = abs(deltaPos[2])

                if not np.all(np.isfinite(deltaPos)):
                    print("invalid delta")
                    validUpdate = False
                    break

                if translation_norm > self.maxTranslation:
                    print("translation step too large")
                    validUpdate = False
                    break

                if rotation_norm > self.maxRotation:
                    print("rotation step too large")
                    validUpdate = False
                    break
                
                updatedVector = self.stateVector + np.append(deltaPos, [0, 0, 0, 0])
                updatedVector[2] = self.wrapAngle(updatedVector[2])
                
                #print("PREVIOUS X: ", self.stateVector[0])
                #print("NEW X: ", updatedVector[0])
                """print("ERROR: ", totalError)
                print("-----------------------------------------------------------")
                print("HESSIAN: ", hessianNorm)
                print("-----------------------------------------------------------")
                print("GRADIENT: ", gradientNorm)
                print("-----------------------------------------------------------")"""
                
                self.stateVector = updatedVector
                iterationCount += 1

                if np.linalg.norm(deltaPos) < 0.001:
                    keepUpdating = False
                    validUpdate = True

            #print("WE HAVE EXITED LOOP")

            if validUpdate and keepUpdating: # Ran out of iterations without settling.
                print("did not converge")
                validUpdate = False

            if validUpdate:
                translationEigenvalues = np.linalg.eigvalsh(hessianNorm[:2, :2]) # Only the x/y block, so the rotation row (different units) doesn't skew the ratio.

                if translationEigenvalues[0] < self.minEigenvalue or translationEigenvalues[0] < self.degenerateRatio * translationEigenvalues[1]:
                    print("scan doesn't constrain translation") # For example sliding along a corridor, or almost nothing contributing.
                    validUpdate = False

                if np.linalg.norm(self.stateVector[:2] - startVector[:2]) > self.maxTotalTranslation: # Every step is capped, but 100 capped steps aren't.
                    print("moved too far from the prediction")
                    validUpdate = False

            if not validUpdate:
                self.stateVector = startVector # Right now a rejected scan leaves the half-updated pose in self.stateVector.

            if validUpdate:
                errorMsg = Float64()
                errorMsg.data = totalError / matchedNodes

                self.error_publisher.publish(errorMsg)

                storageList.extend(self.stateVector.flatten().tolist())
                storageList.extend(np.linalg.pinv(Hessian).flatten().tolist())

                # 2. Package and publish
                msg = Float64MultiArray() 
                msg.data = storageList
                self.correction_publisher.publish(msg)


def main(args = None):
    rclpy.init(args=args)

    newNode = scanMatching()

    rclpy.spin(newNode)

    newNode.destroy_node()

    rclpy.shutdown()

if __name__ == "__main__":
    print("WE RUNNING MAP MATCHING!")
    main()