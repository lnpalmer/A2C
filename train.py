import torch

from models import AtariNet
from envs import create_atari_env, SubprocVecEnv

def train(args, net, optimizer):
    raise NotImplementedError
