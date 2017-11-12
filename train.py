import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fnn
from torch.autograd import Variable

from models import AtariNet
from envs import create_atari_env, SubprocVecEnv

def train(args, net, optimizer, cuda):
    env = SubprocVecEnv([lambda: create_atari_env(args.env_name)] * args.num_workers)
    obs = env.reset()
    net_states = net.make_state(count=args.num_workers)
    if cuda:
        lstm_hs, lstm_cs = net_states
        net_states = lstm_hs.cuda(), lstm_cs.cuda()

    steps = []
    total_steps = 0
    ep_rewards = [0.] * args.num_workers
    render_timer = 0
    while total_steps < args.total_steps:
        for _ in range(args.rollout_steps):
            obs = Variable(torch.from_numpy(obs).float())
            if cuda: obs = obs.cuda()

            policies, values, net_states = net(obs, net_states)

            probs = Fnn.softmax(policies)
            actions = probs.multinomial().data

            obs, rewards, dones, _ = env.step(actions.cpu().numpy())
            obs = env.reset_done()

            # reset the LSTM state for done agents
            masks = (1. - torch.from_numpy(np.array(dones, dtype=np.float32))).unsqueeze(1)
            if cuda: masks = masks.cuda()

            lstm_hs, lstm_cs = net_states
            net_states = lstm_hs * Variable(masks), lstm_cs * Variable(masks)

            rewards = torch.from_numpy(rewards).float().unsqueeze(1)
            if cuda: rewards = rewards.cuda()

            steps.append((rewards, masks, actions, policies, values))

            total_steps += args.num_workers
            for i, done in enumerate(dones):
                ep_rewards[i] += rewards[i]
                if done:
                    print(ep_rewards[i])
                    ep_rewards[i] = 0

            render_timer += 1
            if render_timer == 4:
                env.render(range(args.num_workers))
                render_timer = 0

        final_obs = Variable(torch.from_numpy(obs).float())
        if cuda: final_obs = final_obs.cuda()
        _, final_values, _ = net(final_obs, net_states)
        steps.append((None, None, None, None, final_values))

        actions, policies, values, returns, advantages = process_rollout(args, steps, cuda)

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

        steps = []
        lstm_hs, lstm_cs = net_states
        net_states = Variable(lstm_hs.data), Variable(lstm_cs.data)

def process_rollout(args, steps, cuda):
    _, _, _, _, last_values = steps[-1]
    returns = last_values.data
    advantages = torch.zeros(args.num_workers, 1)
    if cuda: advantages = advantages.cuda()
    out = [None] * (len(steps) - 1)

    for t in reversed(range(len(steps) - 1)):
        rewards, masks, actions, policies, values = steps[t]
        _, _, _, _, next_values = steps[t + 1]

        returns = rewards + returns * args.gamma * masks

        deltas = rewards + next_values.data * args.gamma * masks - values.data
        advantages = advantages * args.gamma * args.lambd * masks + deltas

        out[t] = actions, policies, values, returns, advantages

    return map(lambda x: torch.cat(x, 0), zip(*out))
