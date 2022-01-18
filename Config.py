import  numpy as np



#--------シミュレーション環境関係----------
FIRELD_SIZE= 2
dt = 0.01
CONTROL_STEP = 20 #制御入力を入れるまでの回数(このステップ数は同じ速度で進む)
MAX_STEP = 256*CONTROL_STEP
EPISODE_NUM = 1000
EPISODE_INTERVAL = 100 #ログに吐き出すタイミング
TRAIN_STEP = 10*MAX_STEP #学習を行う頻度．TRAIN_STEPにつき1回学習が起こる
RENDERS = False
NUM_CPU = 8

#--------自作学習用環境関係----------
ROBOT_NUM = 3
OBJECT_NUM = 1
OBSTACLE_NUM = 1
GOAL_NUM = 1
PATHWAY_NUM = 2
ID_NUM = ROBOT_NUM + OBJECT_NUM + OBSTACLE_NUM + GOAL_NUM + PATHWAY_NUM #物体の数
RAY_NUM = 8
OBS_DIM = 4*(ROBOT_NUM-1)+2*OBJECT_NUM+2*OBSTACLE_NUM+2+RAY_NUM #observationの次元 = ニューラルネットへの入力の次元
OUTPUT_DIM = 3 #Actionの次元


#--------学習関係---------------
gamma = 0.99
lamda = 0.95
updates = 2001       # number of updates
epochs = 5          # number of epochs to train the model with sampled data
n_workers = 16    # number of worker processes
worker_steps = 1024 # number of steps to run on each process for a single update
n_mini_batch = 12      # number of mini batches
learning_rate = 1e-3
clip_range = 0.2
save_iteration = 10
entropy_coeff = 0.001

#-------報酬の重み-------------
weight_theta_delta_obj = 0.5/ROBOT_NUM #物体とobjの角度の差に対する重み
weight_x_delta_obj = 0.5/ROBOT_NUM
weight_collision_obj_obs = 0.4 #物体と障害物の接触
weight_collision_obj_pathway = 0.1 #物体とpathwayの接触

foldername = "weight_colliion_obj_obs = 0.4"
