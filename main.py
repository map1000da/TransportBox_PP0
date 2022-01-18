# https://blog.varunajayasiri.com/ml/ppo_pytorch.html
import multiprocessing
import multiprocessing.connection
from typing import Dict, List

import os
import shutil
import glob
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import torch
from torch import optim
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

import matplotlib.pyplot as plt
from Worker import Worker, worker_process
from model import Model

import Config as config

def obs_to_torch(obs: np.ndarray) -> torch.Tensor:
    # scale to `[0, 1]`
    return torch.tensor(obs, dtype=torch.float32, device=device)

class Main:
    def __init__(self):
        # #### Configurations
        self.gamma = config.gamma
        self.lamda = config.lamda
        self.updates = config.updates
        self.epochs = config.epochs
        self.n_workers = config.n_workers
        self.worker_steps = config.worker_steps
        self.n_mini_batch = config.n_mini_batch
        self.obs_dim = config.OBS_DIM
        self.output_dim = config.OUTPUT_DIM

        self.max_mean_reward = -100

        # total number of samples for a single update
        self.batch_size = self.n_workers * self.worker_steps * config.ROBOT_NUM
        # size of a mini batch
        self.mini_batch_size = self.batch_size // self.n_mini_batch
        assert (self.batch_size % self.n_mini_batch == 0)

        # #### Initialize

        # create workers
        self.workers = [Worker(i) for i in range(self.n_workers)]
        self.histories = np.zeros([self.updates, 6]) # reward, std of reward, length, std of length, learnig_rate

        # initialize tensors for observations
        self.obs = np.zeros((self.n_workers, config.ROBOT_NUM, self.obs_dim), dtype=np.float32)
        #self.masks = np.zeros((self.n_workers, NUM_PREY, NUM_N-1, NUM_N-1), dtype=np.uint8)
        for worker in self.workers:
            worker.child.send(("reset", None))

        print("aaaaaaaaaaaaaaaaaaaaaaa\n\n\n")
        #print(self.obs)
        for i, worker in enumerate(self.workers):
            #obs, masks = worker.child.recv()
            #print(i, worker.child.recv())
            self.obs[i] = worker.child.recv()

            #self.masks[i] = masks
        # model for sampling
        #self.model = MultiHeadAdditiveModel(NUM_FLAG, DATA_DIM, NUM_HEAD, DIM_Q, ACTION_DIM).to(device)
        self.model = Model(self.obs_dim, self.output_dim)

        # optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)

        self.dir_path = "./logs/" + config.foldername + "/"

        try:
            os.mkdir(self.dir_path)
        except FileExistsError:
            pass
        try:
            os.mkdir(self.dir_path+"model")
        except FileExistsError:
            pass

        for name in glob.glob('./*.py'):
            shutil.copyfile(f"./{name}", f"{self.dir_path}{name}")


    def sample(self) -> (Dict[str, np.ndarray], List):
        """### Sample data with current policy"""

        rewards_done = []
        lengths_done = []

        rewards = np.zeros((self.n_workers, self.worker_steps, config.ROBOT_NUM), dtype=np.float32)
        actions = np.zeros((self.n_workers, self.worker_steps, config.ROBOT_NUM), dtype=np.int32)
        done = np.zeros((self.n_workers, self.worker_steps, config.ROBOT_NUM), dtype=np.bool_)
        obs = np.zeros((self.n_workers,  self.worker_steps, config.ROBOT_NUM, self.obs_dim), dtype=np.float32)
        #masks = np.zeros((self.n_workers, self.worker_steps, NUM_PREY, NUM_N-1, NUM_N-1), dtype=np.uint8)
        log_pis = np.zeros((self.n_workers, self.worker_steps, config.ROBOT_NUM), dtype=np.float32)
        values = np.zeros((self.n_workers, self.worker_steps, config.ROBOT_NUM), dtype=np.float32)

        # sample `worker_steps` from each worker
        for t in range(self.worker_steps):
            with torch.no_grad():
                # `self.obs` keeps track of the last observation from each worker,
                #  which is the input for the model to sample the next action
                #obs[:, t] = self.obs

                obs[:, t] = self.obs


                # sample actions from $\pi_{\theta_{OLD}}$ for each worker;
                #  this returns arrays of size `n_workers`
                pi, v = self.model(obs_to_torch(self.obs)) #ここ.forward()いらないの？
                values[:, t] = v.cpu().numpy()[:,:,0]
                a = pi.sample()
                actions[:, t] = a.cpu().numpy()
                log_pis[:, t] = pi.log_prob(a).cpu().numpy()

            # run sampled actions on each worker
            for w, worker in enumerate(self.workers):
                worker.child.send(("step", actions[w, t]))

            for w, worker in enumerate(self.workers):
                # get results after executing the actions
                #self.obs[w], self.masks[w], rewards[w, t], done[w, t], info = worker.child.recv()
                self.obs[w] , rewards[w, t], done[w, t], info = worker.child.recv()

                if info:
                    rewards_done.append(info["reward"])
                    lengths_done.append(info["length"])

        # calculate advantages
        advantages = self._calc_advantages(done, rewards, values)
        samples = {
            'obs': obs,
            'actions': actions,
            'values': values,
            'log_pis': log_pis,
            'advantages': advantages
        }

        # samples are currently in [workers, time] table,
        #  we should flatten it
        samples_flat = {}
        for k, v in samples.items():
            if k == 'obs':
                v = v.reshape(v.shape[0] * v.shape[1] * v.shape[2], config.OBS_DIM)
                samples_flat[k] = obs_to_torch(v)
            else:
                v = v.reshape(v.shape[0] * v.shape[1] * v.shape[2], *v.shape[3:])
                samples_flat[k] = torch.tensor(v, device=device)

        return samples_flat, np.array(rewards_done), np.array(lengths_done)

    def _calc_advantages(self, done: np.ndarray, rewards: np.ndarray, values: np.ndarray) -> np.ndarray:

        # advantages table
        advantages = np.zeros((self.n_workers, self.worker_steps, config.ROBOT_NUM), dtype=np.float32)
        last_advantage = 0

        # $V(s_{t+1})$
        _, last_value = self.model(obs_to_torch(self.obs))
        last_value = last_value.cpu().data.numpy()[:,:,0]

        #####
        #####   うまく動いているか謎！！！！！
        #####

        for t in reversed(range(self.worker_steps)):
            # mask if episode completed after step $t$
            mask = 1.0 - done[:, t]
            last_value = last_value * mask
            last_advantage = last_advantage * mask

            delta = rewards[:, t] + self.gamma * last_value - values[:, t]

            last_advantage = delta + self.gamma * self.lamda * last_advantage
            advantages[:, t] = last_advantage

            last_value = values[:, t]

        return advantages

    def train(self, samples: Dict[str, torch.Tensor], learning_rate: float, clip_range: float):
        """
        ### Train the model based on samples
        """

        for _ in range(self.epochs):
            # shuffle for each epoch
            indexes = torch.randperm(self.batch_size)

            # for each mini batch
            for start in range(0, self.batch_size, self.mini_batch_size):
                # get mini batch
                end = start + self.mini_batch_size
                mini_batch_indexes = indexes[start: end]
                mini_batch = {}
                for k, v in samples.items():
                    mini_batch[k] = v[mini_batch_indexes]

                # train
                loss = self._calc_loss(clip_range=clip_range,
                                        samples=mini_batch)

                # compute gradients
                for pg in self.optimizer.param_groups:
                    pg['lr'] = learning_rate
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()

    @staticmethod
    def _normalize(adv: torch.Tensor):
        """#### Normalize advantage function"""
        return (adv - adv.mean()) / (adv.std() + 1e-8)

    def _calc_loss(self, samples: Dict[str, torch.Tensor], clip_range: float) -> torch.Tensor:

        sampled_return = samples['values'] + samples['advantages']

        sampled_normalized_advantage = self._normalize(samples['advantages'])

        # Sampled observations are fed into the model to get $\pi_\theta(a_t|s_t)$ and $V^{\pi_\theta}(s_t)$;
        #  we are treating observations as state
        pi, value = self.model(samples['obs'])

        # #### Policy

        log_pi = pi.log_prob(samples['actions'])

        ratio = torch.exp(log_pi - samples['log_pis'])

        clipped_ratio = ratio.clamp(min=1.0 - clip_range,
                                    max=1.0 + clip_range)
        policy_reward = torch.min(ratio * sampled_normalized_advantage,
                                  clipped_ratio * sampled_normalized_advantage)
        policy_reward = policy_reward.mean()

        # #### Entropy Bonus

        entropy_bonus = pi.entropy()
        entropy_bonus = entropy_bonus.mean()

        # #### Value
        clipped_value = samples['values'] + (value - samples['values']).clamp(min=-clip_range,
                                                                              max=clip_range)
        vf_loss = torch.max((value - sampled_return) ** 2, (clipped_value - sampled_return) ** 2)
        vf_loss = 1 * vf_loss.mean()

        loss = -(policy_reward - vf_loss + config.entropy_coeff * entropy_bonus)

        # for monitoring
        #approx_kl_divergence = .5 * ((samples['log_pis'] - log_pi) ** 2).mean()
        #clip_fraction = (abs((ratio - 1.0)) > clip_range).to(torch.float).mean()

        #tracker.add({'policy_reward': policy_reward,
        #             'vf_loss': vf_loss,
        #             'entropy_bonus': entropy_bonus,
        #             'kl_div': approx_kl_divergence,
        #             'clip_fraction': clip_fraction})

        return loss

    def run_training_loop(self):
        """
        ### Run training loop
        """

        # last 100 episode information
        #tracker.set_queue('reward', 100, True)

        self.model.save(self.dir_path + "model/init_model.pth")


        for update in range(self.updates):
            progress = update / self.updates

            # decreasing `learning_rate` and `clip_range` $\epsilon$
            #学習に合わしてlearnig_rateを減らしているのではなく，一定の割合で減らしているみたいだ
            learning_rate = config.learning_rate * (1 - progress)
            clip_range = config.clip_range * (1 - progress)

            # sample with current policy
            samples, rewards, lengths = self.sample()

            num_game = len(rewards)
            self.histories[update,:] = [
                np.mean(rewards), np.std(rewards),
                np.min(rewards), np.max(rewards), np.mean(lengths), learning_rate
            ]

            print(f"Iteration:{update:<4} reward:{self.histories[update, 0]:<.3}±{self.histories[update,1]:<.3} min:{self.histories[update, 2]:<.3} max:{self.histories[update, 3]:<.3} learning_rate:{self.histories[update, 5]} ")

            # train the model
            self.train(samples, learning_rate, clip_range)


            # write summary info to the writer, and log to the screen
            #tracker.save()
            if (update ) % config.save_iteration == 0:
                self.model.save(self.dir_path + "model/model_{}.pth".format(update))

            #Update the best model
            if np.mean(rewards) > self.max_mean_reward:
                self.model.save(self.dir_path + "model/best_model_{}.pth".format(update))
                self.model.save(self.dir_path + "model/best_model.pth")
                self.max_mean_reward = np.mean(rewards)

            if (update + 1) % 5 == 0:
                self.render(self.histories[:update])
                np.savetxt(self.dir_path + "log.txt", self.histories[:update+1,:],fmt="%.6f",delimiter = ',')
            #    logger.log()

    def render(self, hist):
        plt.cla()
        plt.plot(hist[:,0])
        plt.plot(hist[:,1])
        plt.plot(hist[:,2])
        plt.plot(hist[:,3])

        plt.grid()
        plt.draw()
        plt.pause(0.01)

    def destroy(self):
        """
        ### Destroy
        Stop the workers
        """
        for worker in self.workers:
            worker.child.send(("close", None))

# ## Run it
if __name__ == "__main__":
    #experiment.create()
    print(device)
    m = Main()
    #experiment.start()
    m.run_training_loop()
    m.destroy()
