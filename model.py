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

'''
Basically the model is inspired by the Variational Recurrent Neural Network (VRNN)
from https://arxiv.org/abs/1506.02216.
'''

class Model(nn.Module):
    def __init__(self, x_dim, a_dim, h_dim, z_dim, n_layers, bias=False):
        super(Model, self).__init__()
        self.x_dim = x_dim
        self.a_dim = a_dim
		self.h_dim = h_dim
		self.z_dim = z_dim
		self.n_layers = n_layers

		#feature-extracting transformations
		self.phi_x = nn.Sequential(
			nn.Linear(x_dim, h_dim),
			nn.ReLU(),
			nn.Linear(h_dim, h_dim),
			nn.ReLU()
        )
		self.phi_z = nn.Sequential(
			nn.Linear(z_dim, h_dim),
			nn.ReLU()
        )

		#encoder
		self.enc = nn.Sequential(
			nn.Linear(h_dim + h_dim + a_dim, h_dim),
			nn.ReLU(),
			nn.Linear(h_dim, h_dim),
			nn.ReLU()
        )
		self.enc_mean = nn.Linear(h_dim, z_dim)
		self.enc_std = nn.Sequential(
			nn.Linear(h_dim, z_dim),
			nn.Softplus()
        )

		#prior
		self.prior = nn.Sequential(
			nn.Linear(h_dim + a_dim, h_dim),
			nn.ReLU()
        )
		self.prior_mean = nn.Linear(h_dim, z_dim)
		self.prior_std = nn.Sequential(
			nn.Linear(h_dim, z_dim),
			nn.Softplus()
        )

		#decoder
		self.dec = nn.Sequential(
			nn.Linear(h_dim + h_dim + a_dim, h_dim),
			nn.ReLU(),
			nn.Linear(h_dim, h_dim),
			nn.ReLU(),
            nn.Linear(h_dim, x_dim)
        )

		#recurrence (Policy Network!)
		self.rnn = nn.GRU(h_dim + h_dim, h_dim, n_layers, bias)
        self.policy = nn.Sequential(
            nn.Linear(h_dim, a_dim),
			nn.Tanh()
        )

        self.all_enc_mean = []
        self.all_enc_std = []
        self.all_dec = []
        self.kld_loss = 0
        self.nll_loss = 0
    
    def init_states(self, h=None, prev_a=None):
        if h is None:
            h = torch.zeros(self.n_layers, 1, self.h_dim).requires_grad_()
            prev_a = torch.zeros(1, self.a_dim).requires_grad_()
        else:
            h = h.clone()
            prev_a = prev_a.clone()
        self.all_enc_mean = []
        self.all_enc_std = []
        self.all_dec = []
        self.kld_loss = 0
        self.nll_loss = 0
        return (h, prev_a)
    
    def reset_parameters(self, stdv=1e-1):
		for weight in self.parameters():
			weight.data.normal_(0, stdv)
    
    def forward(self, x_t, h, prev_a):
        '''
        - x: observation vector
        - h: hidden states of policy network
        - prev_a: previous action vector from policy network
        '''
        phi_x_t = self.phi_x(x_t)
        prev_a = prev_a.view(1, -1)

        #encoder
        enc_t = self.enc(torch.cat([phi_x_t, h[-1].view(1, -1), prev_a], 1))
        enc_mean_t = self.enc_mean(enc_t)
        enc_std_t = self.enc_std(enc_t)

        #prior
        prior_t = self.prior(torch.cat([h[-1].view(1, -1), prev_a], 1))
        prior_mean_t = self.prior_mean(prior_t)
        prior_std_t = self.prior_std(prior_t)

        #sampling and reparameterization
        z_t = self._reparameterized_sample(enc_mean_t, enc_std_t)
        phi_z_t = self.phi_z(z_t)

        #decoder
        dec_t = self.dec(torch.cat([phi_z_t, h[-1].view(1, -1), prev_a], 1))

        #recurrence (Policy Network)
        a, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(0), h)
        a = self.policy(a)

        #computing losses
        self.kld_loss += self._kld_gauss(enc_mean_t, enc_std_t, prior_mean_t, prior_std_t)

        self.all_enc_std.append(enc_std_t)
        self.all_enc_mean.append(enc_mean_t)
        self.all_dec_mean.append(dec_t)

        return (dec_t, h, a)
    

    def _reparameterized_sample(self, mean, std):
		"""using std to sample"""
		eps = torch.FloatTensor(std.size()).normal_()
		eps = eps.requires_grad_()
		return eps.mul(std).add_(mean)

	def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
		"""Using std to compute KLD"""

		kld_element =  (2 * torch.log(std_2) - 2 * torch.log(std_1) + 
			(std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
			std_2.pow(2) - 1)
		return	0.5 * torch.sum(kld_element)

	def _nll_bernoulli(self, theta, x):
		return - torch.sum(x*torch.log(theta) + (1-x)*torch.log(1-theta))