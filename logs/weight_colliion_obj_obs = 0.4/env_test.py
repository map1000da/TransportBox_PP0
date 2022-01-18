from myenv.env import MyEnv
import numpy as np

env = MyEnv(renders=1)
#print(env.action_space)
obs = env.reset()
#env.render()
n_steps = 1000
for i in range(n_steps):
  #print(i)
  myaction = [0, 0, 0]

  #print(obs)
  obs, reward, done, info = env.step(myaction)
  print(i, obs)
  #env.render()
  if done:
    print("Goal !!", "reward=", reward)
    obs = env.reset()
