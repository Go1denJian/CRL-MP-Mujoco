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


def get_loss(actor, delta, states, actions):
    mu, std, logstd = actor(torch.Tensor(states))
    log_policy = log_density(torch.Tensor(np.array(actions)), mu, std, logstd)
    delta = delta.unsqueeze(1)

    objective = delta * log_policy
    objective = objective.mean()
    return - objective


def train_critic(critic, states, returns, critic_optim):
    criterion = torch.nn.MSELoss()  # 平方和(损失函数)
    n = len(states)
    arr = np.arange(n)

    for epoch in range(5):
        np.random.shuffle(arr)  # 对列表随机打乱顺序

        for i in range(n // hp.batch_size):
            batch_index = arr[hp.batch_size * i: hp.batch_size * (i + 1)]
            batch_index = torch.LongTensor(batch_index)
            inputs = torch.Tensor(states)[batch_index]
            target = returns.unsqueeze(1)[batch_index]  # 实际值

            values = critic(inputs)  # 预测值
            loss = criterion(values, target)
            critic_optim.zero_grad()
            loss.backward()
            critic_optim.step()


def train_actor(actor, delta, states, actions, actor_optim):
    loss = get_loss(actor, delta, states, actions)
    actor_optim.zero_grad()
    loss.backward()
    actor_optim.step()


def train_model(actor, critic, memory, actor_optim, critic_optim):
    memory = np.array(memory)
    states = np.vstack(memory[:, 0])
    actions = list(memory[:, 1])
    rewards = list(memory[:, 2])
    masks = list(memory[:, 3])
    delta = list(memory[:, 4])

    returns = get_returns(rewards, masks)
    returns_n = (returns - returns.mean())/returns.std()
    train_critic(critic, states, returns_n, critic_optim)
    train_actor(actor, delta, states, actions, actor_optim)
    return returns
