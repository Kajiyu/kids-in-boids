#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys
import random
import numpy as np
from scipy.stats import entropy
import gym
from gym import spaces

################## Global Params ####################################
BOIDS = 10
SPEED_LIMIT = 100
MAX_BOUNDARY = 1000
BOUNDARY_FORCE = 50
DT = 0.07
MODE = "3d"
#####################################################################

def random_start(w_range=50, mode="3d"):
    if mode == "3d":
        return np.array([(random.random()-0.5)*w_range, (random.random()-0.5)*w_range, (random.random()-0.5)*w_range])
    else:
        return np.array([(random.random()-0.5)*w_range, (random.random()-0.5)*w_range])

def limit_speed(boid):
    # Limit boid speed.
    mag_velocity = np.sqrt(boid.velocity.dot(boid.velocity))
    if mag_velocity > SPEED_LIMIT:
        boid.velocity /= boid.velocity/(mag_velocity / SPEED_LIMIT)

def simulate_boundary(boid):
    # Create viewing boundaries.
    for i in range(len(boid.position)):
        if boid.position[i] < (-1)*MAX_BOUNDARY:
            boid.velocity[i] += BOUNDARY_FORCE
        elif boid.position[i] > MAX_BOUNDARY:
            boid.velocity[i] -= BOUNDARY_FORCE

class BoidsEnv(gym.Env):
    def __init__(self):
        super(BoidsEnv).__init__()
        self.mode = MODE
        if self.mode == "3d":
            self.dim = 3
        else:
            self.dim = 2
        self.observation_space = spaces.Box((-1)*MAX_BOUNDARY, MAX_BOUNDARY, shape=(BOIDS, self.dim), dtype=np.int32) # others' positions
        self.action_space = spaces.Box(0., 1., shape=(self.dim,), dtype=np.float32), # velocity
        self.reset()

    def reset(self):
        self.boids = []
        for i in range(BOIDS):
            self.boids.append(Boid(random_start(mode=self.mode), mode=self.mode, dt=DT))
        if self.mode == "3d":
            init_pos = np.array([0,0,0]).astype("float32")
        else:
            init_pos = np.array([0,0]).astype("float32")
        self.kid_agent = BoidKid(init_pos, mode=self.mode, dt=DT)
        self.steps = 0

    def step(self, action):
        simulate_boundary(self.kid_agent)
        reward = self.kid_agent.get_reward(self.boids, action) # TODO: implementing Baysian Surprise.
        for boid in self.boids:
            simulate_boundary(boid)
        for boid in self.boids:
            boid.update_velocity(self.boids+[self.kid_agent])
        for boid in self.boids:
            boid.move()
        action = action * SPEED_LIMIT
        self.kid_agent.move(action)

        observation = []
        for boid in self.boids:
            rel_pos = boid.position - self.kid_agent.position
            observation.append(rel_pos.tolist())
        observation = np.array(observation).astype("int32")
        self.steps =self.steps + 1
        self.done = None
        return observation, reward, self.done, {}

    def render(self, mode='human', close=False):
        pass
    
    def close(self):
        pass


class Boid:
    def __init__(self, init_position, mode="3d", dt=0.05):
        self.mode = mode
        self.dt = dt
        if self.mode == "3d":
            self.velocity = np.array([0,0,0]).astype("float32")
        else:
            self.velocity = np.array([0,0]).astype("float32")
        self.position = init_position

    def update_velocity(self, boids):
        v1 = self.rule1(boids)
        v2 = self.rule2(boids)
        v3 = self.rule2(boids)
        self.__temp = v1 + v2 + v3

    def move(self):
        self.velocity += self.__temp
        limit_speed(self)
        self.position = self.position + (self.velocity * self.dt)

    def rule1(self, boids):
        # clumping
        if self.mode == "3d":
            vector = np.array([0,0,0]).astype("float32")
        else:
            vector = np.array([0,0]).astype("float32")
        for boid in boids:
            if boid is not self:
                vector = vector + boid.position
        vector = vector / float(len(boids) - 1)
        return (vector - self.position) / 7.5

    def rule2(self, boids):
        # avoidance
        if self.mode == "3d":
            vector = np.array([0,0,0]).astype("float32")
        else:
            vector = np.array([0,0]).astype("float32")
        for boid in boids:
            if boid is not self:
                tmp = self.position - boid.position
                if np.sqrt(tmp.dot(tmp)) < 25:
                    vector = vector - (boid.position - self.position)
        return vector

    def rule3(self, boids):
        # schooling
        if self.mode == "3d":
            vector = np.array([0,0,0]).astype("float32")
        else:
            vector = np.array([0,0]).astype("float32")
        for boid in boids:
            if boid is not self:
                vector = vector + boid.velocity
        vector = vector / float(len(boids) - 1)
        return (vector - self.velocity) / 2.0


# Agent for training.(only getting action from NN model)
class BoidKid(Boid):
    def __init__(self, init_position, mode="3d", dt=0.01):
        super(BoidKid, self).__init__(init_position, mode, dt)
    
    def move(self, velocity):
        self.velocity = velocity
        limit_speed(self)
        self.position = self.position + (self.velocity * self.dt)
    
    def get_reward(self, boids, action):
        '''
        a: prediction velocity vector
        b: answer velocity vector
        Reward = dot(a, b) / (|b|)**2 (0 <= Reward <= 1)
        '''
        v1 = self.rule1(boids)
        v2 = self.rule2(boids)
        v3 = self.rule2(boids)
        _v = v1 + v2 + v3
        _v = self.velocity + _v
        flag_vec = action * _v
        flag = True
        for i in range(len(flag_vec)):
            if flag_vec[i] < 0:
                flag = False
        if flag == False:
            reward = 0
        else:
            cos = np.sum(flag_vec) / (np.linalg.norm(action)*np.linalg.norm(_v))
            reward = (np.linalg.norm(action)/np.linalg.norm(_v)) * cos
        print(action, _v)
        return reward
