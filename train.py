import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fnn
from torch.autograd import Variable

from utils import mean_std_groups

def train(args, net, optimizer, env, cuda):
    obs = env.reset()

    if args.plot_reward:
        total_steps_plt = []
        ep_reward_plt = []

    steps = []
    total_steps = 0
    ep_rewards = [0.] * args.num_workers
    render_timer = 0
    plot_timer = 0
    while total_steps < args.total_steps:
        for _ in range(args.rollout_steps):
            obs = Variable(torch.from_numpy(obs.transpose((0, 3, 1, 2))).float() / 255.)
            if cuda: obs = obs.cuda()

            # network forward pass
            policies, values = net(obs)

            probs = Fnn.softmax(policies)
            actions = probs.multinomial().data

            # gather env data, reset done envs and update their obs
            obs, rewards, dones, _ = env.step(actions.cpu().numpy())

            # reset the LSTM state for done envs
            masks = (1. - torch.from_numpy(np.array(dones, dtype=np.float32))).unsqueeze(1)
            if cuda: masks = masks.cuda()

            total_steps += args.num_workers
            for i, done in enumerate(dones):
                ep_rewards[i] += rewards[i]
                if done:
                    if args.plot_reward:
                        total_steps_plt.append(total_steps)
                        ep_reward_plt.append(ep_rewards[i])
                    ep_rewards[i] = 0

            if args.plot_reward:
                plot_timer += args.num_workers # time on total steps
                if plot_timer == 100000:
                    x_means, _, y_means, y_stds = mean_std_groups(np.array(total_steps_plt), np.array(ep_reward_plt), args.plot_group_size)
                    fig = plt.figure()
                    fig.set_size_inches(8, 6)
                    plt.ticklabel_format(axis='x', style='sci', scilimits=(-2, 6))
                    plt.errorbar(x_means, y_means, yerr=y_stds, ecolor='xkcd:blue', fmt='xkcd:black', capsize=5, elinewidth=1.5, mew=1.5, linewidth=1.5)
                    plt.title('Training progress (%s)' % args.env_name)
                    plt.xlabel('Total steps')
                    plt.ylabel('Episode reward')
                    plt.savefig('ep_reward.png', dpi=200)
                    plt.clf()
                    plt.close()
                    plot_timer = 0

            rewards = torch.from_numpy(rewards).float().unsqueeze(1)
            if cuda: rewards = rewards.cuda()

            steps.append((rewards, masks, actions, policies, values))

        final_obs = Variable(torch.from_numpy(obs.transpose((0, 3, 1, 2))).float() / 255.)
        if cuda: final_obs = final_obs.cuda()
        _, final_values = net(final_obs)
        steps.append((None, None, None, None, final_values))

        actions, policies, values, returns, advantages = process_rollout(args, steps, cuda)

        # calculate action probabilities
        probs = Fnn.softmax(policies)
        log_probs = Fnn.log_softmax(policies)
        log_action_probs = log_probs.gather(1, Variable(actions))

        policy_loss = (-log_action_probs * Variable(advantages)).sum()
        value_loss = (.5 * (values - Variable(returns)) ** 2.).sum()
        entropy_loss = (log_probs * probs).sum()

        loss = policy_loss + value_loss * args.value_coeff + entropy_loss * args.entropy_coeff
        loss.backward()

        nn.utils.clip_grad_norm(net.parameters(), args.grad_norm_limit)
        optimizer.step()
        optimizer.zero_grad()

        # cut LSTM state autograd connection to previous rollout
        steps = []

    env.close()

def process_rollout(args, steps, cuda):
    # bootstrap discounted returns with final value estimates
    _, _, _, _, last_values = steps[-1]
    returns = last_values.data

    advantages = torch.zeros(args.num_workers, 1)
    if cuda: advantages = advantages.cuda()

    out = [None] * (len(steps) - 1)

    # run Generalized Advantage Estimation, calculate returns, advantages
    for t in reversed(range(len(steps) - 1)):
        rewards, masks, actions, policies, values = steps[t]
        _, _, _, _, next_values = steps[t + 1]

        returns = rewards + returns * args.gamma * masks

        deltas = rewards + next_values.data * args.gamma * masks - values.data
        advantages = advantages * args.gamma * args.lambd * masks + deltas

        out[t] = actions, policies, values, returns, advantages

    # return data as batched Tensors, Variables
    return map(lambda x: torch.cat(x, 0), zip(*out))
