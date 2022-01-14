# -*-coding:utf-8 -*-
# Author :Yang
# Data:2021/11/15 22:25
import gym
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F



BATCH_SIZE = 64                                # 样本数量
LR = 0.01                                       # 学习率
EPSILON = 0.9                                   # greedy policy
GAMMA = 0.7                                     # reward discount
TARGET_REPLACE_ITER = 100                       # 目标网络更新频率
MEMORY_CAPACITY = 10000                          # 记忆库容量

class Network(nn.Module):
    def __init__(self, input_size, out_putsize, hidden_size=64):
        super(Network, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear1.weight.data.normal_(0, 0.1)  # 权重初始化

        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear2.weight.data.normal_(0, 0.1)  # 权重初始化

        self.linear3 = nn.Linear(hidden_size, out_putsize)
        self.linear2.weight.data.normal_(0, 0.1)  # 权重初始化

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        action_value = F.relu(self.linear3(x))
        return action_value


# 定义经验池
class DQNReplayer:
    def __init__(self, capacity):
        self.memory = pd.DataFrame(index=range(capacity),
                                   columns=['observation', 'action', 'reward', 'next_observation', 'done'])
        self.i = 0  # 索引
        self.count = 0  # 计数
        self.capacity = capacity  # 容量

    def store(self, *args):
        self.memory.loc[self.i] = args
        self.i = (self.i + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def sample(self, size):
        indices = np.random.choice(self.count, size=size)
        return (np.stack(self.memory.loc[indices, field]) for field in self.memory.columns)


class DQNAgent:
    def __init__(self, env, lr=0.001, batch_size=64, gamma=0.7, epsilon=0.01, replayer_capacity=1000):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.replayer_capacity = replayer_capacity
        self.memory_counter = 0
        self.learn_step_counter = 0
        self.eval_net = Network(self.env.observation_space.shape[0], env.action_space.n)
        self.target_net = Network(self.env.observation_space.shape[0], env.action_space.n)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr)
        self.loss_func = nn.MSELoss()
        self.replayer = DQNReplayer(replayer_capacity)

    def decide(self, observation):
        x = torch.unsqueeze(torch.FloatTensor(observation), 0)
        if np.random.rand() < self.epsilon:
            actions_value = self.eval_net.forward(x)  # 通过对评估网络输入状态x，前向传播获得动作值
            action = torch.max(actions_value, 1)[1].data.numpy()  # 输出每一行最大值的索引，并转化为numpy ndarray形式
            action = action[0]
        else:
            action = np.random.randint(0, env.action_space.n)
        return action

    def store_transition(self, observation, action, reward, next_observation, done):
        self.replayer.store(observation, action, reward, next_observation, done)  # 储存经验
        self.memory_counter += 1

    def learn(self):

        if self.learn_step_counter % 100 == 0:  # 一开始触发，然后每100步触发
            self.target_net.load_state_dict(self.eval_net.state_dict())  # 将评估网络的参数赋给目标网络
        self.learn_step_counter += 1

        observations, actions, rewards, next_observations, dones = self.replayer.sample(self.batch_size)  # 拿经验
        s = torch.FloatTensor(observations.reshape(self.batch_size,-1))
        a = torch.LongTensor(actions.reshape(self.batch_size,-1))
        r = torch.FloatTensor(rewards.reshape(self.batch_size,-1))
        ns = torch.FloatTensor(next_observations.reshape(self.batch_size,-1))
        d = torch.FloatTensor(dones.reshape(self.batch_size,-1))#是否结束标志

        q_next = self.target_net(ns).detach()#不对target网络求导
        # next_max_qs = next_qs.max(axis =-1)
        q_target = r + self.gamma * (1. - d) * q_next.max(1)[0].view(self.batch_size , 1)

        q_eval = self.eval_net(s).gather(1, a)

        # q_eval[np.arange( q_target.shape[0],actions)] = q_target
        # q_eval = q_eval[np.arange( q_target.shape[0],actions)]
        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()  # 清空上一步的残余更新参数值
        loss.backward()  # 误差反向传播, 计算参数更新值
        self.optimizer.step()


def play_qlearning(env, agent, render = False, train=False):
    # print('---------------------下一个回合开始-------------------------')

    episode_reward = 0  # 回合总奖励
    observation = env.reset()  # 开始回合

    # print(len(observation))
    while True:

        if render:
            env.render()

        action = agent.decide(observation)  # 从智能体中拿到动作
        next_observation, reward, done, _ = env.step(action)  # 执行动作，得到奖励，与下一个时刻的状态
        #
        agent.store_transition(observation, action, reward, next_observation, done)
        episode_reward += reward  # 计算一个回合的奖励

        if train and (agent.memory_counter > agent.replayer_capacity):  # 是否训练策略/价值网络
             agent.learn()
        if done:
            break
        observation = next_observation
    return episode_reward

if __name__ == '__main__':
    env = gym.make("MountainCar-v0")
    agent = DQNAgent(env)
    episode = 500
    episode_rewards = []
    for e in range(episode):
        print("-----------------{} Episode--------------------".format(e))
        episode_reward = play_qlearning(env, agent, render=True, train=True)

        episode_rewards.append(episode_reward)
        print("reward = {}".format(episode_reward))
        if episode%100 == 0:
            torch.save(agent.eval_net.state_dict(), os.path.join('../02/checkpoint', "model_params"))
    env.close()
    print('------------------done------------------')
    plt.plot(episode_rewards)
    plt.show()
    #修改方案，增加维度