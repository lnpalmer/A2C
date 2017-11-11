import argparse
import torch
import torch.optim as optim

from models import AtariNet
from envs import create_atari_env
from train import train

parser = argparse.ArgumentParser(description='A2C')
parser.add_argument('env_name', type=str)
parser.add_argument('--no-cuda', action='store_true')
parser.add_argument('--num-workers', type=int, default=4)
parser.add_argument('--rollout-steps', type=int, default=20)
parser.add_argument('--total-steps', type=int, default=int(4e7))
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--lambd', type=float, default=1.00)
parser.add_argument('--value_coeff', type=float, default=0.5)
parser.add_argument('--entropy_coeff', type=float, default=0.01)

args = parser.parse_args()

env = create_atari_env(args.env_name)
net = AtariNet(env.action_space.n)
optimizer = optim.Adam(net.parameters(), lr=args.lr)

cuda = torch.cuda.is_available() and not args.no_cuda
if cuda: net = net.cuda()

train(args, net, optimizer, cuda)
