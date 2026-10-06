import rclpy
import time
import numpy as np
import random

from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64MultiArray

class EKF(Node):
    def __init__(self):
        super().__init__("EKF")
        self.stateVector = np.array([0, # x pos 0
                        0, # y pos 1
                        0, # theta 2
                        0, # linear velocity 3
                        0, # angular velocity 4
                        0.1221926719, # linear bias 5; This is preset to start, and is just the value i got from a 2 minute standstill test
                        -1.28 * 10 ** - 7]) # angular bias 6       These values are updated later using the covariance and noise matrixes when we do correction phase.

        self.covarianceMatrix = np.diag([
                            0.12**2,                 # x variance
                            0.12**2,                 # y variance
                            np.deg2rad(3.5)**2,        # theta variance
                            0.12**2,                 # velocity variance
                            0.12**2,                 # angular velocity variance
                            0.025**2,                 # linear bias variance
                            0.025**2                  # angular bias variance
                        ])

        self.HMatrix = np.array([
                        [1, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0, 0]
                    ])
        
        self.sensorNoiseMatrix = np.diag([
            0.010**2,           # x process uncertainty
            0.010**2,           # y process uncertainty
            np.deg2rad(0.38)**2, # theta process uncertainty
            0.05**2,            # velocity uncertainty
            0.05**2,            # angular velocity uncertainty
            1e-5**2,            # linear bias random walk
            1e-5**2             # angular bias random walk
        ])

        self.newOdom = 0
        self.odomVel = 0
        self.odomAngVel = 0

        self.startTime = time.perf_counter() # Represents the time from the start of the program, in ms, to the time that we last updated our self.stateVector. This is used to determine delta T

        self.IMU_subscriber = self.create_subscription(
            Imu,
            '/imu',
            self.updateIMU,
            1
        )

        self.odomSubscriber = self.create_subscription(
            Odometry,
            '/odom',
            self.updateOdom,
            1
        )

        self.NDTSubscriber = self.create_subscription(
            Float64MultiArray,
            '/correction',
            self.correctStateVector,
            1
        )

        self.prediction_publisher = self.create_publisher(
            Float64MultiArray,
            '/prediction',
            10
        )

        self.corrected_publisher = self.create_publisher(
            Float64MultiArray,
            '/corrected',
            10
        )
        

    def updateIMU(self, msg):
        linearAccelX = msg.linear_acceleration.x
        angularVelocity = msg.angular_velocity.z

        self.updateStateVector(time.perf_counter() - self.startTime, angularVelocity, linearAccelX)

    def updateOdom(self, msg):        
        odomVelX = msg.twist.twist.linear.x
        odomVelY = msg.twist.twist.linear.y

        self.odomAngVel = msg.twist.twist.angular.z
        self.newOdom = 1
        self.odomVel = np.sqrt(odomVelX ** 2 + odomVelY ** 2)

    def wrapAngle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def calculateF(self, deltaT, linAcc):
        theta = self.stateVector[2]
        stateLinearVel = self.stateVector[3]
        linVelocity = (stateLinearVel * abs(self.newOdom - 1) + self.newOdom * self.odomVel)
        accBias = self.stateVector[5]
        
        F = np.array([[1, 
              0, 
              -linVelocity * np.sin(theta) * deltaT - (1/2) * (linAcc - accBias) * np.sin(theta) * (deltaT ** 2), 
              np.cos(theta) * deltaT, 
              0,
              -(1/2) * np.cos(theta) * (deltaT ** 2), 
              0],

             [0,
              1,
              linVelocity * np.cos(theta) * deltaT + (1/2) * (linAcc - accBias) * np.cos(theta) * (deltaT ** 2),
              np.sin(theta) * deltaT,
              0,
              -(1/2) * np.sin(theta) * (deltaT ** 2),
              0],

             [0, 0, 1, 0, deltaT, 0, -deltaT],

             [0, 0, 0, 1, 0, -deltaT, 0],

             [0, 0, 0, 0, 1, 0, -deltaT],

             [0, 0, 0, 0, 0, 1, 0],

             [0, 0, 0, 0, 0, 0, 1]])
        
        return F

    def constructW(self): 
        maxWalk = 1e-5**2

        aWalk = random.gauss(0, 1) * maxWalk
        wWalk = random.gauss(0, 1) * maxWalk

        W = np.diag([
            0.01**2,           # x process uncertainty
            0.01**2,           # y process uncertainty
            np.deg2rad(0.15)**2, # theta process uncertainty
            0.02**2,            # velocity uncertainty
            0.02**2,            # angular velocity uncertainty
            maxWalk ** 2,       # Linear bias
            maxWalk ** 2        # Angular bias
        ])

        return W
    
    

    def calculateQ(self, deltaT, theta):
        W = self.constructW()

        gLin = [(1/2) * np.cos(theta) * deltaT ** 2, (1/2) * np.sin(theta) * deltaT ** 2, 0, deltaT, 0, 1, 1]
        gAng = [0, 0, deltaT, 0, 0, 1, 1]

        qLin = np.outer(gLin, gLin.T) @ W 
        qAng = np.outer(gAng, gAng.T) @ W
        
        if self.newOdom:
            gOdom = [np.cos(theta) * deltaT, np.sin(theta) * deltaT, 0, 1, 0, 1, 1]
            qOdom = np.outer(gOdom, gOdom.T) @ W

            return qOdom + qLin + qAng

        else: 
            return qLin + qAng

    def updateStateVector(self, deltaT, angularVel, linearAccel): # accelerations come from IMU, and angular velocity comes from IMU
        xPos = self.stateVector[0]
        yPos = self.stateVector[1]
        theta = self.stateVector[2]
        stateLinearVel = self.stateVector[3]
        stateAngularVel = self.stateVector[4]
        linearBias = self.stateVector[5]
        angularBias = self.stateVector[6]

        # Makes linearVel either the stored state velocity, or the velocity outputted by the odometry if there is a new velocity available.
        linearVel = (stateLinearVel * abs(self.newOdom - 1) + self.newOdom * self.odomVel)

        if self.odomVel < 0.001 and self.newOdom == 1:
            # We are definitively stopped. Kill the velocity and ignore noisy acceleration.
            linearVel = 0
            linearAccel = linearBias # Zero out the effective acceleration
        
        if self.odomAngVel > 0.001:
            angularVel = (stateAngularVel * abs(self.newOdom - 1) + self.newOdom * self.odomAngVel - angularBias)
        else:
            angularVel = angularVel - angularBias

        self.stateVector[0] = xPos + linearVel * np.cos(theta) * deltaT + (1/2) * (linearAccel - linearBias) * np.cos(theta) * (deltaT ** 2)
        self.stateVector[1] = yPos + linearVel * np.sin(theta) * deltaT + (1/2) * (linearAccel - linearBias) * np.sin(theta) * (deltaT ** 2)
        self.stateVector[2] = self.wrapAngle(theta + angularVel * deltaT)
        self.stateVector[3] = linearVel + (linearAccel - linearBias) * deltaT
        self.stateVector[4] = angularVel
        # self.stateVector values 5 and 6 are not updated in this prediction phase. They are updated during the propagation of error in correction phase.

        F = self.calculateF(deltaT, linearAccel)
        Q = self.sensorNoiseMatrix #self.calculateQ(deltaT, theta)

        self.covarianceMatrix = F @ self.covarianceMatrix @ F.T + Q

        msg = Float64MultiArray()
        msg.data = self.stateVector.flatten().tolist()
        print("predicted X: ", self.stateVector[0])
        print("Predicted Y: ", self.stateVector[1])
        print("Preidcted Rotation: ", self.stateVector[2])

        self.prediction_publisher.publish(msg)
        self.startTime = time.perf_counter()
        self.newOdom = 0

    def correctStateVector(self, message):
        lidarVector = np.zeros(7) 
        lidarUncertainty = [[], [], []]
        counter = 0
        uncertaintyCounter = 0
        
        # Convert the incoming message directly into a numpy array
        data_array = np.array(message.data)
        
        # Slice the first 7 elements for the state vector
        lidarVector = data_array[:7]
        
        # Slice the remaining 9 elements and reshape them back into a 3x3 matrix
        lidarUncertainty = data_array[7:16].reshape((3, 3))

        #print("LIDARVECTOR: ", lidarVector)
        #print("IDAR UNCERTAINTY: ", lidarUncertainty)

        residualVector = lidarVector[:3] - self.stateVector[:3]
        residualVector[2] = self.wrapAngle(residualVector[2])

        innovationCovariance = self.HMatrix @ self.covarianceMatrix @ self.HMatrix.T + lidarUncertainty
        
        kalmanGain = self.covarianceMatrix @ self.HMatrix.T @ np.linalg.inv(innovationCovariance)

        #print("KALMAN GAIN: ", kalmanGain)
        #print("RESIDUAL VECTOR: ", residualVector)

        self.stateVector = self.stateVector.T + kalmanGain @ residualVector.T
        self.stateVector[2] = self.wrapAngle(self.stateVector[2])

        self.covarianceMatrix = (np.identity(7) - kalmanGain @ self.HMatrix) @ self.covarianceMatrix @ (np.identity(7) - kalmanGain @ self.HMatrix).T + kalmanGain @ lidarUncertainty @ kalmanGain.T

        msg = Float64MultiArray()
        msg.data = self.stateVector.flatten().tolist()

        self.corrected_publisher.publish(msg)
        print("CORRECTED x: ", self.stateVector[0])
        print("corrected y: ", self.stateVector[1])
        print("corrected rotation: ", self.stateVector[2])  

def main(args = None):
    rclpy.init(args=args)

    newNode = EKF()

    rclpy.spin(newNode)

    newNode.destroy_node()

    rclpy.shutdown()

if __name__ == "__main__":
    main()
    print("WE running EKF")