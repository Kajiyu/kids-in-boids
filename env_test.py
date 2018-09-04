#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys
import gym
import random
import numpy as np
import gym.spaces
import env

env = gym.make('Boids3d-v0')
env.reset()

for i in range(10000):
    observation, reward, done, _ = env.step(np.array([random.random()*0.1, random.random()*0.1, random.random()*0.1]))
    print("Observation: ", observation)
    print("Reward: ", reward)