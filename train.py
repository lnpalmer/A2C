import torch
import torch.nn as nn
import torch.nn.functional as Fnn
from torch.autograd import Variable

from models import AtariNet
from envs import create_atari_env, SubprocVecEnv

def train(args, net, optimizer):
    env = SubprocVecEnv([lambda: create_atari_env(args.env_name)] * args.num_workers)
    obs = env.reset()
    net_states = net.make_state(count=args.num_workers)

    steps = []
    total_steps = 0
    while total_steps < args.total_steps:
        for _ in range(args.rollout_steps):
            obs = Variable(torch.from_numpy(obs).float())
            obs_in = obs

            policies, values, net_states = net(obs, net_states)

            probs = Fnn.softmax(policies)
            actions = probs.multinomial().data

            obs, rewards, dones, _ = env.step(actions.numpy())
            obs = env.reset_done()

            rewards = torch.from_numpy(rewards).float()
            steps.append((rewards, dones, actions, policies, values))

        final_obs = Variable(torch.from_numpy(obs).float())
        _, final_values, _ = net(final_obs, net_states)
        steps.append((None, None, None, None, final_values))

        actions, policies, values, returns, advantages = process_rollout(args, steps)

        probs = Fnn.softmax(policies)
        log_probs = Fnn.log_softmax(policies)
        log_action_probs = log_probs.gather(1, Variable(actions))

        policy_loss = (-log_action_probs * Variable(advantages)).sum()
        value_loss = (.5 * (values - Variable(returns)) ** 2.).sum()
        entropy_loss = (log_probs * probs).sum()

        loss = policy_loss + value_loss * args.value_coeff + entropy_loss * args.entropy_coeff
        loss.backward()

        nn.utils.clip_grad_norm(net.parameters(), 40.)
        optimizer.step()
        optimizer.zero_grad()

        print(returns.sum())

        steps = []
        lstm_hs, lstm_cs = net_states
        net_states = Variable(lstm_hs.data), Variable(lstm_cs.data)

def process_rollout(args, steps):
    _, _, _, _, last_values = steps[-1]
    returns = last_values.data
    advantages = torch.zeros(args.num_workers, 1)
    out = [None] * (len(steps) - 1)

    for t in reversed(range(len(steps) - 1)):
        rewards, dones, actions, policies, values = steps[t]
        _, _, _, _, next_values = steps[t + 1]

        returns = rewards + returns * args.gamma

        deltas = rewards + next_values.data * args.gamma - values.data
        advantages = advantages * args.gamma * args.lambd + deltas

        out[t] = actions, policies, values, returns, advantages

    return map(lambda x: torch.cat(x, 0), zip(*out))
