import warnings
warnings.simplefilter('ignore', FutureWarning)
warnings.simplefilter('ignore', Warning)
import gym
import os
import numpy as np
import datetime
import pytz
import torch
from model import Model
from myenv.env import MyEnv
import matplotlib.pyplot as plt
import Config.config as config
import matplotlib.animation as animation
from PIL import Image
import cv2

device = torch.device("cpu")
def obs_to_torch(obs: np.ndarray) -> torch.Tensor:
    # scale to `[0, 1]`
    return torch.tensor(obs, dtype=torch.float32, device=device)

def pil2cv(image):
    ''' PIL型 -> OpenCV型 '''
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image

#--------グローバル変数------------
MAX_STEP = config.MAX_STEP
SAVE_MOVIE = False

env = MyEnv(renders=1)
model = Model(config.OBS_DIM, config.OUTPUT_DIM)
model.load(f"logs/{config.foldername}/model/model_40.pth")

fig = plt.figure()
ims = []
# 10回試行する
for i in range(5):
    obs = env.reset()
    step = 0
    while True:
        pi, _ = model(obs_to_torch(obs))

        probs = pi.probs
        probs = probs.detach().numpy()
        #print(probs)
        actions = np.argmax(probs,axis=1)
        #actions = pi.sample().detach().numpy()

        if SAVE_MOVIE:
            #動画の作成
            img = env.getCameraImage()

            im = Image.fromarray(img) #np ⇛ Pillow
            im = im.convert("RGB") #RGBa ⇛　RGB
            im = pil2cv(im) #pillow ⇛　cv2
            #print("shape:", im.shape)
            ims.append(im)


        obs, rewards, dones, info = env.step(actions)
        #print(step, env.goal_min)

        #print("step:", step)
        step += 1
        #env.render()
        if dones:
            print(info["length"],info["reward"])
            break

if SAVE_MOVIE:
    fourcc = cv2.VideoWriter_fourcc('m','p','4', 'v')
    video  = cv2.VideoWriter('ImgVideo.mp4', int(fourcc), 30, (640, 640))
    for i in range(len(ims)):
        cv2.imwrite("result/cap_{}.png".format(i), ims[i])
        #print(i)
        video.write(ims[i])

    video.release()
