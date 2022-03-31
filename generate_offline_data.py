import gym
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace as SN
import pickle
import os

import numpy as np
from networks.drqn import Q_net


class EpisodeBatch(dict):
    def __init__(self, obs, action, reward, done):
        super().__init__()
        self["obs"] = obs
        self["action"] = action
        self["reward"] = reward
        self["done"] = done
        self.batch_size = len(self.get("obs"))

        self._longest_episode()

    def __repr__(self):
        return f"EpisodeBatch(batch_size={len(self.get('obs'))}, maxlen={self.maxlen}, fields=(obs, action, reward, done))"

    def _longest_episode(self):
        self.maxlen = (
            np.amax(np.argmax(self.get("done"), axis=1)) + 1
        )  # for slicing issues
        self["maxlen"] = self.maxlen  # just for easy consistent access


class SequenceReplayMemory:
    def __init__(self, args):
        """
        Probably don't need a new_state memory, but see how.
        """
        self.args = args
        self.mem_size = args.mem_size
        self.mem_ctr = 0
        self.state_memory = np.zeros(
            (self.mem_size, args.maxlen, args.input_shape), dtype=np.float32
        )
        self.action_memory = np.zeros((self.mem_size, args.maxlen), dtype=np.int64)
        self.reward_memory = np.zeros((self.mem_size, args.maxlen), dtype=np.float32)
        # ones to store dones, since shorter episodes are still done
        self.terminal_memory = np.ones((self.mem_size, args.maxlen), dtype=np.bool)

    def store_episode(self, episode):
        index = self.mem_ctr % self.mem_size
        obs = np.array(episode["obs"])
        actions = np.array(episode["action"])
        rewards = np.array(episode["reward"])
        dones = np.array(episode["done"])

        self.state_memory[index, : obs.shape[0], :] = obs
        self.action_memory[index, : actions.shape[0]] = actions
        self.reward_memory[index, : rewards.shape[0]] = rewards
        self.terminal_memory[index, : dones.shape[0]] = dones

        self.mem_ctr += 1

    def sample_episodes(self, batch_size):
        max_mem = min(self.mem_ctr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        # note, these are batches of episodic data
        obs = self.state_memory[batch]
        action = self.action_memory[batch]
        reward = self.reward_memory[batch]
        done = self.terminal_memory[batch]

        ep_batch = EpisodeBatch(obs, action, reward, done)

        return ep_batch

    def can_sample(self, batch_size):
        return self.mem_ctr >= batch_size


if __name__ == "__main__":
    args = SN()
    args.episodes = 1e6
    args.use_trained = True
    args.save_loc = "./buffer/trained_experience.pkl"
    args.game = "CartPole-v0"
    args.mem_size = 3e6

    env = gym.make(args.game)

    # agent related params
    args.hidden_space = 64
    args.tran1_dim = 256
    args.tran2_dim = 128
    args.action_space = 1 if args.game == "CartPole-v1" else env.action_space.n
    args.projection_out_dim = 16
    args.prediction_dim = 32
    args.grad_norm_clip = 10
    args.batch_size = 64
    args.lr = 5e-5
    args.tau = 1e-2
    args.spr_weight = 0.001
    args.k_steps = 3
    args.use_spr = True
    args.loss_fn = "L1"
    args.device = th.device("cuda:1")

    args.input_shape = env.observation_space.shape[0]
    args.maxlen = 200

    if args.use_trained:
        agent = Q_net(
            args,
            state_space=env.observation_space.shape[0],
            action_space=env.action_space.n,
        ).to(args.device)

    replay_memory = SequenceReplayMemory(args)

    for ep in range(args.episodes):
        ep_info = {"obs": [], "reward": [], "done": [], "action": []}
        env.reset()

        done = False
        while not done:
            if args.use_trained:
                pass
            else:
                action = env.action_space.sample()
                observation, reward, done, _ = env.step(action)
            ep_info["obs"].append(observation)
            ep_info["reward"].append(reward)
            ep_info["done"].append(done)
            ep_info["action"].append(action)

        replay_memory.store_episode(ep_info)

    if not os.path.isdir("./buffer"):
        os.makedirs("./buffer/")

    with open(args.save_loc, "wb") as outfile:
        pickle.dump(replay_memory, outfile)
    print("Successfully written replay buffer!")
