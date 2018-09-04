#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from gym.envs.registration import register
from .environment import *

register(
    id='Boids3d-v0',
    entry_point='boids_env.environment:BoidsEnv',
)