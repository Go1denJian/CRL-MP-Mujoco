import argparse
import gym
import numpy as np
from hopper_env import HopperEnv

import torch
import torch.optim as optim

from model import Actor, Critic
from utils import get_action
from collections import deque
from running_state import ZFilter
from hparams import HyperParams as hp
from PG_MP import train_model

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.90, metavar='G',
                    help='discount factor (default: 0.90)')
parser.add_argument('--seed', type=int, default=500, metavar='N',
                    help='random seed (default: 500)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--selfmade', type=bool, default=True,
                    help='environment with self-made (default: False)')
args = parser.parse_args()

if args.selfmade:
    env = HopperEnv(ctrl_cost_weight=1)
else:
    env = gym.make("Hopper-v3")

env.reset(seed=args.seed)
torch.manual_seed(args.seed)
dtype = torch.float32
device = torch.device("cpu")

num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]

print('state size:', num_inputs)
print('action size:', num_actions)

# 建立网络
actor = Actor(num_inputs, num_actions)
actor_optim = optim.Adam(actor.parameters(), lr=hp.actor_lr)

running_state = ZFilter((num_inputs,), clip=5)
episodes = 0
dataset = {"episodes": [], "score": [], "cost": [], "lagrange": [], "V0": [], "V1": [], "mu": []}
min_V1 = 0

for iter in range(10000):
    actor.eval() # 禁用 BatchNormalization 和 Dropout
    memory = deque()  # 双向队列

    steps = 0
    scores = []
    while steps < 2048:
        episodes += 1
        state = env.reset()
        state = running_state(state)
        score = np.zeros(2)
        for j in range(10000):
            if args.render and episodes % 50 == 0:
                env.render()

            steps += 1
            mu, std, _ = actor(torch.Tensor(state).unsqueeze(0))
            action = get_action(mu, std)[0]
            next_state, reward0, done, _ = env.step(action)
            reward1 = _["ctrl_cost"]
            next_state = running_state(next_state)
            reward = np.array([reward0, reward1])

            if done:
                mask = 0
            else:
                mask = 1

            memory.append([state, action, reward, mask])

            score += reward
            state = next_state

            if done:
                break
        scores.append(score)
    scores = np.array(scores)
    score_avg = np.mean(scores, axis=0)

    dataset['episodes'].append(episodes)
    dataset['score'].append(score_avg[0])
    dataset['cost'].append(-score_avg[1])
    dataset['lagrange'].append(actor.lagrange)

    actor.train()  # 起用 BatchNormalization 和 Dropout
    memory = np.array(memory)
    returns = train_model(actor, memory, actor_optim)

    masks = memory[:, 3]
    V0 = returns[..., 0][masks == 0].mean()
    V1 = returns[..., 1][masks == 0].mean()

    dataset['V0'].append(V0.item())
    dataset['V1'].append(V1.item())
    dataset['mu'].append(actor.mu)
    print('{} episode score is {:.2f}, cost is {:.2f}, lambda is {:.2f}, mu is {:.2f}, iter is {}'.format(
        episodes, score_avg[0], -score_avg[1], dataset['lagrange'][-1], dataset['mu'][-1], iter))
    print('    V0 is {:.2f}, V1 is {:.2f} (average for all)'.format(V0, V1))

    if 2 * actor.mu * (V1 - actor.c1) - actor.lagrange > 0:
        V1 = actor.lagrange / 2 / actor.mu
    else:
        V1 = V1 - actor.c1

    actor.lagrange = max(0.0, actor.lagrange - 2 * actor.mu * V1)
    # min_V1 = abs(V1)

np.save("H_PG_MP.npy", dataset)
# dataset = np.load('H_PG_MP.npy', allow_pickle='TRUE')
torch.save(actor.state_dict(), 'H_PG_MP_actor.pt')

