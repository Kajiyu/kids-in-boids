#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys
import random
import numpy as np

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import Model

import gym
import gym.spaces
import boids_env


TOTAL_EPISODE = 10000
TOTAL_STEP_PER_EPISODE = 100000
OPTIMIZING_SPAN = 30


if __name__ == "__main__":
    # env
    env = gym.make('Boids3d-v0')

    #hyperparameters
    x_dim = env.dim * env.num_boids
    a_dim = env.dim
    h_dim = 60
    z_dim = 16
    n_layers = 2
    clip = 10
    learning_rate = 1e-3
    seed = 128

    #manual seed
    torch.manual_seed(seed)

    model = Model(x_dim, a_dim, h_dim, z_dim, n_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for idx_episode in range(TOTAL_EPISODE):
        optimizer.zero_grad()
        observation, reward, done, _ = env.reset()
        observation = observation.reshape((1, -1))
        x = torch.from_numpy(observation)
        x = x.requires_grad_()
        obs_list = []
        dec_x_list = []
        (h, prev_a) = model.init_states()
        for idx_step in range(TOTAL_STEP_PER_EPISODE):
            (dec_x, h, prev_a) = model(x, h, prev_a)
            observation, reward, done, _ = env.step(prev_a.cpu().numpy().reshape((-1)))
            observation = observation.reshape((1, -1))
            x = torch.from_numpy(observation)
            x = x.requires_grad_()
            obs_list.append(x)
            dec_x_list.append(dec_x)
            if idx_step != 0 and idx_step % OPTIMIZING_SPAN == 0:
                BCE = None
                for j in range(len(obs_list)):
                    if BCE is None:
                        BCE = F.binary_cross_entropy(dec_x_list[j], obs_list[j], size_average=False)
                    else:
                        BCE = BCE + F.binary_cross_entropy(dec_x_list[j], obs_list[j], size_average=False)
                loss = BCE + model.kld_loss
                loss.backward()
                obs_list = []
                dec_x_list = []
                optimizer.zero_grad()
                (h, prev_a) = model.init_states(h, prev_a)