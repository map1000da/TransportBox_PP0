import numpy as np

class BulletObject:
    def __init__(self, _p, urdfFileName, initialPos, initialOrientaion=[0,0,0,1]):
        self.p = _p
        self.objId = self.p.loadURDF(f"./urdfs/{urdfFileName}.urdf", basePosition=initialPos, baseOrientation=initialOrientaion)

    def getJointVelocity(self, index):
        return self.p.getJointState(self.objId, index)[1]

    @property
    def pos(self):
        pos, _ = self.p.getBasePositionAndOrientation(self.objId)
        return np.array(pos)

    @property
    def distance(self):
        pos = np.array(self.pos)
        pos[2] = 0
        return np.linalg.norm(pos)


    @property
    def phi(self):
        _, ori = self.p.getBasePositionAndOrientation(self.objId)
        ori = self.p.getEulerFromQuaternion(ori)
        return ori[2]

    @property
    def v(self):
        return self.p.getBaseVelocity(self.objId)[0]

    @property
    def vNorm(self):
        return np.linalg.norm(self.v)

    @property
    def vNormXY(self):
        v = np.array(self.v)
        v[2] = 0
        return np.linalg.norm(v)

    def getDistance(self, other):
        return np.linalg.norm(self.getToVector(other))

    def getToVector(self, other):
        pos1 = np.array(self.pos)
        pos2 = np.array(other.pos)

        # value of z-axis to zero
        pos1[2] = 0
        pos2[2] = 0

        return pos2-pos1

    def getContactPoints(self, other):
        return self.p.getContactPoints(self.objId, other.objId)

    def isCollision(self, other):
        return self.getContactPoints(other) != ()

class Robot(BulletObject):
    def __init__(self, id, _p, urdfFileName, initialPos=[0,0,0.05], initialOrientaion=[0,0,0,1]):
        super().__init__(_p, urdfFileName, initialPos, initialOrientaion)
        self.id = id
        self.contact = False


    def setMotorSpeed(self, speed):
        left = speed[0]
        right = speed[1]
        maxForce = 0.4
        self.p.setJointMotorControl2(bodyUniqueId=self.objId,
            jointIndex = 3,
            controlMode = self.p.VELOCITY_CONTROL,
            targetVelocity = left,
            force = maxForce)
        self.p.setJointMotorControl2(bodyUniqueId=self.objId,
            jointIndex=1,
            controlMode=self.p.VELOCITY_CONTROL,
            targetVelocity = right,
            force = maxForce)

class Object(BulletObject):
    def __init__(self, p, urdfFileName, initialPos=[0,0,0.05]):
        super().__init__(p, urdfFileName, initialPos)

class Goal(BulletObject):
    def __init__(self, p, urdfFileName, initialPos=[0,0,0.05]):
        super().__init__(p, urdfFileName, initialPos)

class Obstacle(BulletObject):
    def __init__(self, p, urdfFileName, initialPos=[0,0,0.05]):
        super().__init__(p, urdfFileName, initialPos)

class PathWay(BulletObject):
    def __init__(self, id, _p, urdfFileName, initialPos=[0,0,0.05], initialOrientaion=[0,0,0,1]):
        super().__init__(_p, urdfFileName, initialPos, initialOrientaion)
        self.id = id
