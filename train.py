#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys
import random
import argparse
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
OPTIMIZING_SPAN = 5
SAVE_SPAN = 500

bce_loss = nnloss = nn.MSELoss()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cuda', help='gpu mode', action='store_true')
    args = parser.parse_args()

    if args.cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

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

    model = Model(x_dim, a_dim, h_dim, z_dim, n_layers, device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for idx_episode in range(TOTAL_EPISODE):
        model.reset_parameters()
        optimizer.zero_grad()
        observation, reward, done, _ = env.reset()
        observation = observation.reshape((1, -1)).astype("float32")
        x = torch.tensor(observation, device=device).requires_grad_()
        obs_list = []
        dec_x_list = []
        (h, prev_a) = model.init_states()
        for idx_step in range(TOTAL_STEP_PER_EPISODE):
            (dec_x, h, prev_a) = model(x, h, prev_a)
            observation, reward, done, _ = env.step(prev_a.detach().cpu().numpy().reshape((-1)))
            observation = observation.reshape((1, -1)).astype("float32")
            x = torch.tensor(observation, device=device).requires_grad_()
            obs_list.append(x.detach())
            dec_x_list.append(dec_x)
            if idx_step != 0 and idx_step % OPTIMIZING_SPAN == 0:
                MSE = None
                for j in range(len(obs_list)):
                    if MSE is None:
                        MSE = bce_loss(dec_x_list[j], obs_list[j])
                    else:
                        MSE = MSE + bce_loss(dec_x_list[j], obs_list[j])
                loss = MSE + model.kld_loss
                print("Episode", idx_episode, "- Step", idx_step, loss.data[0])
                loss.backward(retain_graph=True)
                optimizer.step()
                nn.utils.clip_grad_norm(model.parameters(), clip)
                obs_list = []
                dec_x_list = []
                optimizer.zero_grad()
                (h, prev_a) = model.init_states(h, prev_a)
            if idx_step % SAVE_SPAN == 0:
                torch.save(model.state_dict(), "./data/model_"+str(idx_episode)+".pth")
                print("----step "+str(idx_step)+" ----")
                print("Kid Position:", env.kid_agent.position)
                print("Kid Velocity:", env.kid_agent.velocity)
                print("Observation:", observation.reshape((-1,3)))
                print("----------------------------")
