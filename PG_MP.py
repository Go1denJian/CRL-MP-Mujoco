import numpy as np
import torch
from hparams import HyperParams as hp
from utils import log_density


def get_returns(rewards, masks):
    rewards = torch.Tensor(np.array(rewards))
    masks = torch.Tensor(masks)
    returns = torch.zeros_like(rewards)

    running_returns = 0

    for t in reversed(range(0, len(rewards))):
        running_returns = rewards[t] + hp.gamma * running_returns * masks[t]
        returns[t] = running_returns

    return returns


def get_loss(actor, returns, states, actions):
    mu, std, logstd = actor(torch.Tensor(states))
    log_policy = log_density(torch.Tensor(np.array(actions)), mu, std, logstd)
    returns = returns.unsqueeze(1)

    objective = returns * log_policy
    objective = objective.mean()
    return - objective


def train_actor(actor, returns, states, actions, actor_optim):
    loss = get_loss(actor, returns, states, actions)
    actor_optim.zero_grad()
    loss.backward()
    actor_optim.step()


def train_model(actor, memory, actor_optim, critic_optim):
    memory = np.array(memory)
    states = np.vstack(memory[:, 0])
    actions = list(memory[:, 1])
    rewards = list(memory[:, 2])
    masks = list(memory[:, 3])

    returns = get_returns(rewards, masks)
    r_V0 = returns[..., 0]
    r_V1 = returns[..., 1] - actor.c1
    zeror = torch.zeros_like(r_V0)
    r_V1 = - (torch.maximum(zeror, actor.lagrange - 2 * actor.mu * (
                r_V1 - actor.c1)) ** 2 - actor.lagrange ** 2) / 4 / actor.mu

    returns_L = r_V0+r_V1
    returns_L = (returns_L - returns_L.mean()) / returns_L.std()

    train_actor(actor, returns_L, states, actions, actor_optim)
    return returns
