#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys
import gym
import random
import numpy as np
import gym.spaces

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from environment import *


def boids_test():
    boids = []
    for i in range(BOIDS):
        boids.append(Boid(random_start()))
    for i in range(1000):
        fig = plt.figure(figsize=(10,7))
        ax = Axes3D(fig)
        # print(i, "round")
        for boid in boids:
            simulate_boundary(boid)
        for boid in boids:
            boid.update_velocity(boids)
        for boid in boids:
            boid.move()
        #     print(boid.position)
        # print("\n\n")
        for boid in boids:
            ax.scatter(boid.position[0], boid.position[1], boid.position[2])
        ax.view_init(30, 30)
        ax.set_xlim3d([-1000, 1000])
        ax.set_ylim3d([-1000, 1000])
        ax.set_zlim3d([-1000, 1000])
        plt.savefig('./png/'+str(i)+'.png')