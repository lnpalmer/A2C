import argparse
import torch
import torch.optim as optim

from models import AtariNet
from envs import create_atari_env
from train import train

parser = argparse.ArgumentParser(description='A2C (Advantage Actor-Critic)')
parser.add_argument('env_name', type=str, help='Gym environment id')
parser.add_argument('--no-cuda', action='store_true', help='use to disable available CUDA')
parser.add_argument('--num-envs', type=int, default=4, help='number of parallel environments')
parser.add_argument('--rollout-steps', type=int, default=20, help='steps per rollout')
parser.add_argument('--total-steps', type=int, default=int(4e7), help='total number of steps to train for')
parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.99, help='gamma parameter for GAE')
parser.add_argument('--lambd', type=float, default=1.00, help='lambda parameter for GAE')
parser.add_argument('--value_coeff', type=float, default=0.5, help='value loss coeffecient')
parser.add_argument('--entropy_coeff', type=float, default=0.01, help='entropy loss coeffecient')

args = parser.parse_args()

env = create_atari_env(args.env_name)
net = AtariNet(env.action_space.n)
optimizer = optim.Adam(net.parameters(), lr=args.lr)

cuda = torch.cuda.is_available() and not args.no_cuda
if cuda: net = net.cuda()

train(args, net, optimizer, cuda)
