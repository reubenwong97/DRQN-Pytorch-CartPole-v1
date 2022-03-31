import sys
from typing import Dict, List, Tuple
from types import SimpleNamespace

import gym
import collections
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

from networks.drqn import Q_net
from utils.network_utils import soft_copy, hard_copy, freeze_params
from networks.spr_networks import TransitionModel, PredictionHead, ProjectionHead
from utils.losses import cosine_loss, L1_loss
from memory.replay_buffer import EpisodeBuffer, EpisodeMemory


def spr_train(
    # q_net,
    # soft_q_net,
    projection,
    soft_projection,
    predictor,
    transition_model,
    hiddens,
    targets,
    actions,
    loss_function,
    args,
    k_steps=2,
):
    # NOTE: I'd rather pass in the targets, do it from outside? X: not feasible, need to time skip
    # soft_copy(soft_q_net, q_net, args.tau)
    # freeze_params(soft_q_net)
    # soft_copy(soft_projection, projection)
    # freeze_params(soft_projection)

    spr_loss = torch.tensor([0.0]).to(args.device)
    # DEBUG: hiddens are [64, 8, 64] and actions are [64, 8, 1] offending index 0 but its the last one that mismatches...
    stacked = torch.cat([hiddens, actions], 2)  # concat along time

    for t in range(1, k_steps + 1):
        next_encoding = transition_model(stacked)
        if t == 1:
            pass  # don't slice if t = 1, want full tensor
        else:
            next_encoding = next_encoding[:, : -(t - 1)]  # discard from the back
        online_projection = projection(next_encoding)
        online_prediction = predictor(online_projection)

        # targets can be done outside since the encoder model does not change
        target_encoding = targets[
            :, t - 1 :
        ]  # t-1 slicing since targets generated from next_observations
        target_projection = soft_projection(target_encoding)

        # currently online_prediction is [64, 7, 16] and target_projection is [64, 8, 16]
        spr_loss_ = loss_function(online_prediction, target_projection)
        spr_loss += spr_loss_

    spr_loss /= k_steps

    return spr_loss


def train(
    q_net=None,
    target_q_net=None,
    projection=None,
    soft_projection=None,
    predictor=None,
    transition_model=None,
    args=None,
    episode_memory=None,
    device=None,
    optimizer=None,
    batch_size=1,
    gamma=0.99,
):

    assert device is not None, "None Device input: device should be selected."

    # NOTE: No need separate soft copy Q-net because current RL implementation
    #      already makes use of soft copying at the same frequency we require
    soft_copy(soft_projection, projection, args.tau)

    # Get batch from replay buffer
    samples, seq_len = episode_memory.sample()

    observations = []
    actions = []
    rewards = []
    next_observations = []
    dones = []

    for i in range(batch_size):
        observations.append(samples[i]["obs"])
        actions.append(samples[i]["acts"])
        rewards.append(samples[i]["rews"])
        next_observations.append(samples[i]["next_obs"])
        dones.append(samples[i]["done"])

    observations = np.array(observations)
    actions = np.array(actions)
    rewards = np.array(rewards)
    next_observations = np.array(next_observations)
    dones = np.array(dones)

    observations = torch.FloatTensor(observations.reshape(batch_size, seq_len, -1)).to(
        device
    )
    actions = torch.LongTensor(actions.reshape(batch_size, seq_len, -1)).to(device)
    rewards = torch.FloatTensor(rewards.reshape(batch_size, seq_len, -1)).to(device)
    next_observations = torch.FloatTensor(
        next_observations.reshape(batch_size, seq_len, -1)
    ).to(device)
    dones = torch.FloatTensor(dones.reshape(batch_size, seq_len, -1)).to(device)

    # h_target, c_target = target_q_net.init_hidden_state(batch_size=batch_size, training=True)
    h_target = target_q_net.init_hidden_state(batch_size=batch_size, training=True)

    q_target, h_target, _ = target_q_net(next_observations, h_target.to(device))

    q_target_max = q_target.max(2)[0].view(batch_size, seq_len, -1).detach()
    targets = rewards + gamma * q_target_max * dones

    h = q_net.init_hidden_state(batch_size=batch_size, training=True)
    q_out, h_ts, final_h = q_net(observations, h.to(device))
    q_a = q_out.gather(2, actions)

    # Multiply Importance Sampling weights to loss
    loss = F.smooth_l1_loss(q_a, targets)

    if args.loss_fn == "cosine":
        loss_func = cosine_loss
    elif args.loss_fn == "L1":
        loss_func = L1_loss
    else:
        raise ValueError("SPR Loss function must be specified")

    if args.use_spr:
        spr_loss = spr_train(
            projection,
            soft_projection,
            predictor,
            transition_model,
            h_ts,
            h_target,
            actions,
            loss_func,
            args,
            args.k_steps,
        )

        loss = loss + args.spr_weight * spr_loss

    # Update Network
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if args.use_spr:
        return loss.item(), spr_loss.item()
    else:
        return loss.item(), 0.0


def seed_torch(seed):
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def save_model(model, path="default.pth"):
    torch.save(model.state_dict(), path)


if __name__ == "__main__":

    env_name = "CartPole-v1"
    # Set gym environment
    env = gym.make(env_name)

    # Set parameters
    args = SimpleNamespace()
    args.hidden_space = 64
    args.tran1_dim = 256
    args.tran2_dim = 128
    args.action_space = 1 if env_name == "CartPole-v1" else env.action_space.n
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
    args.device = torch.device("cuda:1")

    buffer_len = int(100000)
    min_epi_num = 64  # Start moment to train the Q network
    episodes = 650
    print_per_iter = 20
    target_update_period = 4
    eps_start = 0.6
    eps_end = 0.001
    eps_decay = 0.995
    tau = 1e-2
    max_step = 2000

    # DRQN param
    random_update = True  # If you want to do random update instead of sequential update
    lookup_step = 15  # If you want to do random update instead of sequential update
    max_epi_len = 128
    max_epi_step = max_step
    # Env parameters
    model_name = f"DRQN_SPR_{str(args.k_steps)}_steps"
    seed = 1
    exp_num = "SEED" + "_" + str(seed)
    suffix = f"eps_06_sprweight_001_{args.loss_fn}"

    if torch.cuda.is_available():
        device = torch.device("cuda:1")

    # Set the seed
    np.random.seed(seed)
    random.seed(seed)
    seed_torch(seed)
    env.seed(seed)

    # default `log_dir` is "runs" - we'll be more specific here
    writer = SummaryWriter(
        "runs/" + env_name + "_" + model_name + "_" + exp_num + "_" + suffix
    )

    # projection,
    # soft_projection,
    # predictor,

    # Create Q functions
    Q = Q_net(
        args,
        state_space=env.observation_space.shape[0],
        action_space=env.action_space.n,
    ).to(device)
    Q_target = Q_net(
        args,
        state_space=env.observation_space.shape[0],
        action_space=env.action_space.n,
    ).to(device)
    #! NOTE: To remove, since they also use soft updates
    # soft_target = Q_net(
    #     state_space=env.observation_space.shape[0], action_space=env.action_space.n
    # ).to(
    #     device
    # )  # for soft-copying in spr training

    freeze_params(Q_target)
    Q_target.load_state_dict(Q.state_dict())

    # Projection networks
    projector = ProjectionHead(args)
    soft_projector = ProjectionHead(args)
    freeze_params(soft_projector)

    # Predictor
    predictor = PredictionHead(args)

    # Transition model
    transition_model = TransitionModel(args.action_space, args)

    # Set optimizer
    score = 0
    score_sum = 0
    params = list(Q.parameters())
    params += projector.parameters()
    params += predictor.parameters()
    params += transition_model.parameters()
    optimizer = optim.Adam(params, lr=args.lr)

    epsilon = eps_start

    episode_memory = EpisodeMemory(
        random_update=random_update,
        max_epi_num=100,
        max_epi_len=600,
        batch_size=args.batch_size,
        lookup_step=lookup_step,
    )

    # Train
    for i in range(episodes):
        losses = []
        spr_losses = []
        s = env.reset()
        obs = s  # Use only Position of Cart and Pole
        done = False

        episode_record = EpisodeBuffer()
        h = Q.init_hidden_state(batch_size=args.batch_size, training=False)

        for t in range(max_step):

            # Get action
            a, h = Q.sample_action(
                torch.from_numpy(obs).float().to(device).unsqueeze(0).unsqueeze(0),
                h.to(device),
                epsilon,
            )

            # Do action
            s_prime, r, done, _ = env.step(a)
            obs_prime = s_prime

            # make data
            done_mask = 0.0 if done else 1.0

            episode_record.put([obs, a, r / 100.0, obs_prime, done_mask])

            obs = obs_prime

            score += r
            score_sum += r

            if len(episode_memory) >= min_epi_num:
                # collecting loss for logging
                loss, spr_loss = train(
                    Q,
                    Q_target,
                    projector,
                    soft_projector,
                    predictor,
                    transition_model,
                    args,
                    episode_memory,
                    device,
                    optimizer=optimizer,
                    batch_size=args.batch_size,
                )

                if spr_loss != 0.0:
                    spr_losses.append(spr_loss)

                # function already returns the float
                losses.append(loss)

                if (t + 1) % target_update_period == 0:
                    # Q_target.load_state_dict(Q.state_dict()) <- navie update
                    for target_param, local_param in zip(
                        Q_target.parameters(), Q.parameters()
                    ):  # <- soft update
                        target_param.data.copy_(
                            tau * local_param.data + (1.0 - tau) * target_param.data
                        )

            if done:
                break

        episode_memory.put(episode_record)

        epsilon = max(eps_end, epsilon * eps_decay)  # Linear annealing

        if i % print_per_iter == 0 and i != 0:
            print(
                "n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                    i, score_sum / print_per_iter, len(episode_memory), epsilon * 100
                )
            )
            score_sum = 0.0
            save_model(Q, model_name + "_" + exp_num + ".pth")

        # Log the reward
        writer.add_scalar("Rewards per episodes", score, i)
        writer.add_scalar("Loss per episode", np.mean(losses), i)
        writer.add_scalar("SPR Loss per episode", np.mean(spr_losses), i)

        score = 0

    # f1 = plt.figure()
    # plt.plot(np.arange(len(losses)), losses, label="loss")
    # plt.xlabel("Episodes")
    # plt.ylabel("Loss")
    # plt.savefig(f"./plots/spr_rl/k_{args.k_steps}_loss_{args.loss_fn}")

    writer.close()
    env.close()
