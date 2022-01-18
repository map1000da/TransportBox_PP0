import pybullet as p2
import pybullet_data
import pybullet_utils.bullet_client as bc
import time
import numpy as np
import random

import sys
sys.path.append('../')
import Config as config
from robot import Robot, Object, Goal, Obstacle, PathWay

#--------グローバル変数------------
dt = config.dt
CONTROL_STEP = config.CONTROL_STEP #制御入力を入れるまでの回数(このステップ数は同じ速度で進む)
ID_NUM = config.ID_NUM #物体の数
ROBOT_NUM = config.ROBOT_NUM
MAX_STEP = config.MAX_STEP
EPISODE_NUM = config.EPISODE_NUM
EPISODE_INTERVAL = config.EPISODE_INTERVAL #ログに吐き出すタイミング
TRAIN_STEP = config.TRAIN_STEP #学習を行う頻度．TRAIN_STEPにつき1回学習が起こる
RENDERS = config.RENDERS
OBS_DIM = config.OBS_DIM #NNの入力の次元

def rot(x, theta): #回転行列をかけた値を返す
    R = np.array([[np.cos(theta), np.sin(theta)],
                  [-np.sin(theta),  np.cos(theta)]])

    return  R@(x.T)

class MyEnv():

    Actions = [
        [4*np.pi, 4*np.pi], #直進
        [-4*np.pi, 4*np.pi], #約45度左折
        [4*np.pi, -4*np.pi] #約45度右折
    ]

    def __init__(self, renders=0):

        self.robot_num = ROBOT_NUM
        self.pathway_num = config.PATHWAY_NUM
        self.step_count = 0
        self._renders = True if renders==1 else False
        print("render:", self._renders, config.RENDERS)
        self.dt = dt
        self._physics_client_id = -1
        self.action = [0, 0, 0]
        self.max_step = MAX_STEP
        self.collision = None
        self.episode_num = 0
        self.global_step = 0
        self.control_step = CONTROL_STEP
        self.robot_list = []
        self.pathway_list = []
        self.obs_dim = OBS_DIM
        self.ray_num = config.RAY_NUM
        self.ray_length = 10

        self.robot_start_pos = [[-2.3, 1, 0], [-2.3, 1.5, 0], [-2.3, 2, 0]]
        self.robot_orn = [[0 ,0, np.pi/2], [0, 0, np.pi/2], [0, 0 ,np.pi/2]] #初期の向き

        self.object_start_pos = [-random.uniform(1,1.5),random.uniform(1.25,1.75), 0.1]
        self.obstacle_start_pos = [[0, 1.5, 0.1]]
        self.goal_start_pos = [2, 1.5, 0.1]

        self.pathway_pos = [[0, 0, 0],[0, 3, 0]]
        self.pathway_orn = [[0, 0, np.pi/2], [0, 0, np.pi/2]]

        n_actions = len(MyEnv.Actions)

        #self.reset()

        #self.action_space = spaces.Discrete(n_actions)
        #self.observation_space = {agent: spaces.Box(low=0, high=self.field_size, shape=(3*ID_NUM+2,), dtype=np.float32) for agent in self.possible_agents}


        #行動空間(Boxは連続値)
        #self.action_space = spaces.Discrete(n_actions)
        #観測値の空間
        #self.observation_space = spaces.Box(low=0, high=self.field_size, shape=(3*ID_NUM+2,), dtype=np.float32)


    def reset(self):
        """
        #doneがTrueになったとき呼び出されるメソッド
        #返り値はobservarion
        #つまり，何をどのような形で観測値にするかで学習が変わってくる
        """
        #print("ReseT!!")
        self.rewards = [] #1エピソードの報酬を記録
        self.reward = 0 #1エピソードの合計の報酬
        self.step_count = 0

        self.collision = None
        self.goal_min = np.sqrt((self.object_start_pos[0] - self.goal_start_pos[0])**2 + (self.object_start_pos[1] - self.goal_start_pos[1])**2)
        self.goal_min = round(self.goal_min, 3)
        #self.goal_manhattan_x = abs(self.goal_start_pos[0] - self.object_start_pos[0])
        #self.goal_manhattan_x = round(self.goal_manhattan_x, 3)
        #self.goal_manhattan_y = abs(self.goal_start_pos[1] - self.object_start_pos[1])
        #self.goal_manhattan_y = round(self.goal_manhattan_y, 3)
        self.robot_object_start = []
        for i in range(self.robot_num):
            self.robot_object_start.append(round((self.object_start_pos[0] - self.robot_start_pos[i][0])**2 + (self.object_start_pos[1] - self.robot_start_pos[i][1])**2, 3))
        self.episode_num += 1
        #print(self.robot_object_start)
        #print("episode_num:", self.episode_num)

        #ロボットの初期姿勢のランダム化
        self.robot_orn = [[0,0,random.uniform(-np.pi,np.pi)] for _ in range(self.robot_num)]


        if self._physics_client_id < 0:
            print("siaiaia")
            if self._renders:
                print("guuuuuuuuuu")
                self._p = bc.BulletClient(connection_mode=p2.GUI)
                #self._p = bc.BulletClient(connection_mode=p2.DIRECT)
            else:
                self._p = bc.BulletClient()
            self._physics_client_id = self._p._client
            #print("aaaaa:", self._physics_client_id)

            p = self._p
            p.resetSimulation()
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            for i in range(self.robot_num):
                orn = p.getQuaternionFromEuler(self.robot_orn[i])
                self.robot_list.append(Robot(i, p, "robot", self.robot_start_pos[i], orn))
            self.planeID = p.loadURDF("plane.urdf")
            self.object = Object(p, "object", self.object_start_pos)
            self.obstacle = Obstacle(p, "obstacle", random.choice(self.obstacle_start_pos))
            self.goal = Goal(p, "goal", self.goal_start_pos)
            for i in range(self.pathway_num):
                orn = p.getQuaternionFromEuler(self.pathway_orn[i])
                self.pathway_list.append(PathWay(i, p, "pathway", self.pathway_pos[i], orn))

            #self.objectID = p.loadURDF("urdfs/object.urdf", basePosition=self.object_start_pos)
            #self.goalID = p.loadURDF("urdfs/goal.urdf", basePosition=self.goal_start_pos)
            p.setGravity(0, 0, -9.8)
            p.setTimeStep(self.dt)
            for robot in self.robot_list:
                robot.setMotorSpeed([0, 0])
        else:
            #print("unnnch")
            p = self._p
            #p.resetSimulation()
            #p.resetBasePositionAndOrientation(self.robotID, self.robot_start_pos, p.getQuaternionFromEuler([0,0,0]))
            for i in range(self.robot_num):
                #print(self.robot_list[i].objId, self.robot_start_pos[i])
                p.resetBasePositionAndOrientation(self.robot_list[i].objId, self.robot_start_pos[i], p.getQuaternionFromEuler(self.robot_orn[i]))
            p.resetBasePositionAndOrientation(self.object.objId, self.object_start_pos, p.getQuaternionFromEuler([0,0,0]))
            p.resetBasePositionAndOrientation(self.obstacle.objId, random.choice(self.obstacle_start_pos), p.getQuaternionFromEuler([0,0,0]))
            p.resetBasePositionAndOrientation(self.goal.objId, self.goal_start_pos, p.getQuaternionFromEuler([0,0,0]))
            #for i in range(self.pathway_num):
            #    p.resetBasePositionAndOrientation(self.pathway_list[i].id, self.pathway_pos[i], p.getQuaternionFromEuler(self.pathway_orn[i]))

            for robot in self.robot_list:
                robot.setMotorSpeed([0, 0])

        #しばらくシミュレーションをすすめる
        for t in range(self.control_step):
            p.stepSimulation()

        #observation = {self.agents[i]: np.array(np.hstack([self.robot_start_pos[i], np.cos(self.robot_list[i].phi), np.sin(self.robot_list[i].phi), self.box_start_pos, self.goal_start_pos])) for i in range(len(self.agents))}

        obs = self.observe(p)
        return obs

    def step(self, actions):
        """
        #1stepで起こること
        #受け取ったactionに対する評価値rewardを計算し，次の時間の状態(observation)を返す
        #doneでゲームのフラグ管理
        #infoはまじでよく分からん
        #返り値はobservation, reward, done, info
        """
        p = self._p
        reward = 0
        r_trans = 0
        r_obstacle = 0
        r_penalty = 0
        done = False

        #actionの反映
        for i, robot in enumerate(self.robot_list):
            action = actions[i]
            action = MyEnv.Actions[action]
            robot.setMotorSpeed(action)

        #しばらくシミュレーションをすすめる
        for t in range(self.control_step):
            p.stepSimulation()
            self.step_count += 1
            self.global_step += 1


        #---------------------報酬計算---------------------------
        #---------------------■1 r_trans-----------------------
        #①物体をゴールに輸送できたかの報酬(ユークリッド距離)
        object_goal_dict = np.sqrt((self.object.pos[0] - self.goal.pos[0])**2 + (self.object.pos[1] - self.goal.pos[1])**2)
        object_goal_dict = round(object_goal_dict, 4)
        if object_goal_dict < self.goal_min:
            r_trans += (self.goal_min - object_goal_dict)
            self.goal_min = object_goal_dict
            #print("接近")

        #②ロボット達が物体方向を向いているか
        for i, robot in enumerate(self.robot_list):
            vec = self.object.pos - robot.pos
            theta_obj = np.arctan2(vec[0], vec[1])
            theta_delta = abs(theta_obj - robot.phi)
            r_trans -= (config.weight_theta_delta_obj*theta_delta/np.pi)/(self.max_step/self.control_step)

        #③ロボット達が物体方向に近づいたかの報酬
        for i, robot in enumerate(self.robot_list):
            robot_object_dict = (self.object.pos[0] - robot.pos[0])**2 + (self.object.pos[1] - robot.pos[1])**2
            robot_object_dict = round(robot_object_dict, 4)
            r_trans -= (config.weight_x_delta_obj*min(1, robot_object_dict/self.robot_object_start[i]))/(self.max_step/self.control_step)

        #---------------------■2 r_obstacle---------------------------------

        #---------------------■3 r_penalty------------------------------------
        #①障害物と物体の接触に関する報酬
        if len(p.getContactPoints(self.object.objId, self.obstacle.objId)) > 0:
            r_penalty -= config.weight_collision_obj_obs/(self.max_step/self.control_step)


        #②物体とpathwayの衝突に関する報酬
        for i in range(len(self.pathway_list)):
            if len(p.getContactPoints(self.object.objId, self.pathway_list[i].objId)) > 0:
                r_penalty -= config.weight_collision_obj_pathway/(self.max_step/self.control_step)
                break


        #物体とゴールの接触の確認
        if len(p.getContactPoints(self.object.objId, self.goal.objId)) > 0:
            done = True
            #r_trans += 3
            print("Goal!!")
            print("episode:{} rewards:{}".format(self.episode_num, self.reward))
        reward = r_trans + r_obstacle + r_penalty
        self.reward += reward


        #返り値の計算
        obs = self.observe(p)

        if self.step_count > self.max_step:
            #print("episode_end")
            #print("episode:{} rewards:{}".format(self.episode_num, self.reward))
            done = True

        self.rewards.append(reward)
        if done:
            info = {"reward": sum(self.rewards),
                        "length": len(self.rewards)}
            obs = self.reset()
        else:
            info = None
        return obs, reward, done, info

    def render(self):
        pass

    def observe(self, p): #入力相対座標の旧バージョン
        obs = np.zeros([self.robot_num, self.obs_dim])
        for i, robot in enumerate(self.robot_list):
            #my_obs = np.hstack([robot.pos[:2], np.cos(robot.phi), np.sin(robot.phi)]) #[自分の位置，姿勢]
            other_obs = []
            for j, other_robot in enumerate(self.robot_list):
                if i == j:
                    continue
                else:
                    other_obs.extend(rot(other_robot.pos[:2] - robot.pos[:2], robot.phi))
                    other_obs.append(np.cos(other_robot.phi - robot.phi))
                    other_obs.append(np.sin(other_robot.phi- robot.phi))
            other_robot_obs = np.array(other_obs)
            #temp = np.hstack([my_obs, other_robot_obs]) #[自分の位置，姿勢，他のロボットの位置，姿勢]
            object_pos = []
            object_pos.extend(rot(self.object.pos[:2] - robot.pos[:2], robot.phi))
            obstacle_pos = []
            object_pos.extend(rot(self.obstacle.pos[:2] - robot.pos[:2], robot.phi))
            object_pos.extend(obstacle_pos)
            other_pos = np.hstack([object_pos, rot(self.goal.pos[:2] - robot.pos[:2], robot.phi)]) #[物体の位置/姿勢, 障害物の位置/姿勢, ゴールの位置]
            #print("other_pos:", other_pos)
            temp_v2 = np.hstack([other_robot_obs, other_pos]) #[自分の位置/姿勢，他のロボットの位置/姿勢, 物体の位置/姿勢, 障害物の位置/姿勢, ゴールの位置]

            #自分のロボットから各方向へのpathwayとの距離
            distance_to_pathway = []
            for j in range(config.RAY_NUM):
                rayFromPosition = [0]*3
                rayFromPosition[0] = self.robot_list[i].pos[0]
                rayFromPosition[1] = self.robot_list[i].pos[1]
                rayFromPosition[2] = self.robot_list[i].pos[2] + 0.3
                rayToPosition = [0]*3
                rayToPosition[0] = self.robot_list[i].pos[0] + self.ray_length*np.cos(self.robot_list[i].phi + j*2*np.pi/config.RAY_NUM)
                rayToPosition[1] = self.robot_list[i].pos[1] + self.ray_length*np.sin(self.robot_list[i].phi + j*2*np.pi/config.RAY_NUM)
                rayToPosition[2] = self.robot_list[i].pos[2] + 0.3
                #----render raycast--------
                #p.addUserDebugLine(rayFromPosition, rayToPosition, lifeTime=2)
                a = p.rayTest(rayFromPosition, rayToPosition)
                distance_to_pathway.append(self.ray_length*a[0][2])
            distance_to_pathway = np.array(distance_to_pathway)
            #print(obs[i], distance_to_pathway)
            obs[i] = np.hstack([temp_v2, distance_to_pathway]) #[自分の位置/姿勢，他のロボットの位置/姿勢, 物体の位置/姿勢, 障害物の位置/姿勢, ゴールの位置, 4方向のpathwayの距離]


        return obs

    def observe_old(self, p): #入力絶対座標の旧バージョン
        obs = np.zeros([self.robot_num, self.obs_dim])
        for i, robot in enumerate(self.robot_list):
            my_obs = np.hstack([robot.pos[:2], np.cos(robot.phi), np.sin(robot.phi)]) #[自分の位置，姿勢]
            other_obs = []
            for j, other_robot in enumerate(self.robot_list):
                if i == j:
                    continue
                else:
                    other_obs.extend(other_robot.pos[:2])
                    other_obs.append(np.cos(other_robot.phi))
                    other_obs.append(np.sin(other_robot.phi))
            other_robot_obs = np.array(other_obs)
            temp = np.hstack([my_obs, other_robot_obs]) #[自分の位置，姿勢，他のロボットの位置，姿勢]
            object_pos = []
            object_pos.extend(self.object.pos[:2])
            object_pos.append(np.cos(self.object.phi))
            object_pos.append(np.sin(self.object.phi))
            obstacle_pos = []
            object_pos.extend(self.obstacle.pos[:2])
            object_pos.append(np.cos(self.obstacle.phi))
            object_pos.append(np.sin(self.obstacle.phi))

            object_pos.extend(obstacle_pos)
            other_pos = np.hstack([object_pos, self.goal.pos[:2]]) #[物体の位置/姿勢, 障害物の位置/姿勢, ゴールの位置]
            #print("other_pos:", other_pos)
            temp_v2 = np.hstack([temp, other_pos]) #[自分の位置/姿勢，他のロボットの位置/姿勢, 物体の位置/姿勢, 障害物の位置/姿勢, ゴールの位置]

            #自分のロボットから各方向へのpathwayとの距離
            distance_to_pathway = []
            for j in range(config.RAY_NUM):
                rayFromPosition = [0]*3
                rayFromPosition[0] = self.robot_list[i].pos[0]
                rayFromPosition[1] = self.robot_list[i].pos[1]
                rayFromPosition[2] = self.robot_list[i].pos[2] + 0.3
                rayToPosition = [0]*3
                rayToPosition[0] = self.robot_list[i].pos[0] + self.ray_length*np.cos(self.robot_list[i].phi + j*np.pi/2)
                rayToPosition[1] = self.robot_list[i].pos[1] + self.ray_length*np.sin(self.robot_list[i].phi + j*np.pi/2)
                rayToPosition[2] = self.robot_list[i].pos[2] + 0.3
                a = p.rayTest(rayFromPosition, rayToPosition)
                distance_to_pathway.append(self.ray_length*a[0][2])
            distance_to_pathway = np.array(distance_to_pathway)
            #print(obs[i], distance_to_pathway)
            obs[i] = np.hstack([temp_v2, distance_to_pathway]) #[自分の位置/姿勢，他のロボットの位置/姿勢, 物体の位置/姿勢, 障害物の位置/姿勢, ゴールの位置, 4方向のpathwayの距離]


        return obs

    def getCameraImage(self):
        proj_matrix = self._p.computeProjectionMatrixFOV(
			fov=30, #カメラfov(視野？)
            aspect=320/320, #カメラのアスペクト比
			nearVal=1,
            farVal=10.0
            )
        view_matrix = self._p.computeViewMatrix(
            cameraEyePosition = [2,2,10], #カメラの物理的な位置
            cameraTargetPosition = [2,2,0], #カメラを向けたいポイント
            cameraUpVector = [0,1,0] #カメラの上部を刺すベクトル
            )

        """
        view_matrix = self._p.computeViewMatrixFromYawPitchRoll(
			cameraTargetPosition=[4/2,4/2,0],
			distance=9,
			yaw=0,
			pitch=-90,
			roll=0,
			upAxisIndex=2)
        """

        (_, _, px, _, _)  = self._p.getCameraImage(640, 640,view_matrix,proj_matrix,  renderer=self._p.ER_TINY_RENDERER)
        #(_, _, px, _, _)  = self._p.getCameraImage(640, 640)
        img = np.reshape(px, (640, 640, 4))

        return img


#env = MyEnv()
#print(env.action_space.shape)
