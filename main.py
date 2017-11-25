import argparse
import torch
import torch.optim as optim
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_frame_stack import VecFrameStack

from models import AtariCNN
from envs import make_env, RenderSubprocVecEnv
from train import train

parser = argparse.ArgumentParser(description='A2C (Advantage Actor-Critic)')
parser.add_argument('env_name', type=str, help='Gym environment id')
parser.add_argument('--no-cuda', action='store_true', help='use to disable available CUDA')
parser.add_argument('--num-workers', type=int, default=4, help='number of parallel workers')
parser.add_argument('--rollout-steps', type=int, default=20, help='steps per rollout')
parser.add_argument('--total-steps', type=int, default=int(4e7), help='total number of steps to train for')
parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.99, help='gamma parameter for GAE')
parser.add_argument('--lambd', type=float, default=1.00, help='lambda parameter for GAE')
parser.add_argument('--value_coeff', type=float, default=0.5, help='value loss coeffecient')
parser.add_argument('--entropy_coeff', type=float, default=0.01, help='entropy loss coeffecient')
parser.add_argument('--grad_norm_limit', type=float, default=40., help='gradient norm clipping threshold')
parser.add_argument('--render', action='store_true', help='render training environments')
parser.add_argument('--render-interval', type=int, default=4, help='number of steps between environment renders')
parser.add_argument('--plot-reward', action='store_true', help='plot episode reward vs. total steps')
parser.add_argument('--plot-group-size', type=int, default=80, help='number of episodes grouped into a single plot point')
parser.add_argument('--seed', type=int, default=0, help='random seed')

args = parser.parse_args()

env_fns = []
for rank in range(args.num_workers):
    env_fns.append(lambda: make_env(args.env_name, rank, args.seed + rank))
if args.render:
    venv = RenderSubprocVecEnv(env_fns, args.render_interval)
else:
    venv = SubprocVecEnv(env_fns)
venv = VecFrameStack(venv, 4)

net = AtariCNN(venv.action_space.n)
optimizer = optim.Adam(net.parameters(), lr=args.lr)

cuda = torch.cuda.is_available() and not args.no_cuda
if cuda: net = net.cuda()

train(args, net, optimizer, venv, cuda)
