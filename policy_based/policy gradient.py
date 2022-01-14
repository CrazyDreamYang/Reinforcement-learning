import torch
import gym
import numpy as np
import os
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical

#定义超参数
#gamma = 0.99 #衰减程度
#seed = 456 #随机种子数

# env = gym.make('CartPole-v1')
# env.seed(seed)
# torch.manual_seed(seed)    # 策略梯度算法方差很大，设置seed以保证复现性
# print('observation space:',env.observation_space)
# print('action space:',env.action_space)

#np.random.seed(1)
#定义网络，输入是观测空间维度，输出是动作空间维度
class Network(nn.Module):
    def __init__(self, input_size, out_putsize, hidden_size=128):
        super(Network, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear1.weight.data.normal_(0, 0.3)  # 权重初始化

        # self.linear2 = nn.Linear(hidden_size, hidden_size)
        # self.linear2.weight.data.normal_(0, 0.1)  # 权重初始化

        self.linear3 = nn.Linear(hidden_size, out_putsize)
        self.linear3.weight.data.normal_(0, 0.3)  # 权重初始化

    def forward(self, x):
        x = F.relu(self.linear1(x))
        #x = F.relu(self.linear2(x))
        action_scores = self.linear3(x)
        return action_scores#对于离散动作输出的是概率

class PGAgent(object):
    def __init__(self,  env, lr =0.01, gamma = 0.99):
        self.lr = lr
        self.GAMMA = gamma
        self.policy_net = Network(env.observation_space.shape[0],env.action_space.n)

        #定义储存动作，状态、奖励的列表
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []
        self.optimizer =torch.optim.Adam(self.policy_net.parameters(),lr = lr)

        self.time_step = 0

    #储存动作，状态、奖励
    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def decide(self,state):
        #state = torch.from_numpy(state).float().unsqueeze(0)
        observation = torch.FloatTensor(state)#转化为张量，为什么不用添加一个维度
        network_output =  self.policy_net.forward(observation)
        with torch.no_grad():
            prob_weights = F.softmax(network_output, dim=0).numpy()
        # prob_weights = F.softmax(network_output, dim=0).detach().numpy()
        action = np.random.choice(range(prob_weights.shape[0]), p=prob_weights)  # select action w.r.t the actions prob
        return action

    def learn(self):
        self.time_step += 1

        # Step 1: 计算每一步的状态价值
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        # 注意这里是从后往前算的，所以式子还不太一样。算出每一步的状态价值
        # 前面的价值的计算可以利用后面的价值作为中间结果，简化计算；从前往后也可以
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.GAMMA + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        discounted_ep_rs -= np.mean(discounted_ep_rs)  # 减均值
        discounted_ep_rs /= np.std(discounted_ep_rs)  # 除以标准差
        discounted_ep_rs = torch.FloatTensor(discounted_ep_rs)

        # Step 2: 前向传播
        softmax_input =  self.policy_net.forward(torch.FloatTensor(self.ep_obs))
        # all_act_prob = F.softmax(softmax_input, dim=0).detach().numpy()
        neg_log_prob = F.cross_entropy(input=softmax_input, target=torch.LongTensor(self.ep_as), reduction='none')

        # Step 3: 反向传播
        loss = torch.mean(neg_log_prob * discounted_ep_rs)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 每次学习完后清空数组
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []


def playPG(env, agent, step, train = True, render = False):
    #运行一个回合,学习的时候，一个回合学习一次，使用的是均值更新
    episode_reward = 0  # 回合总奖励
    state= env.reset()  # 开始回合
    STEP = 0
    while True:

        if render:
           env.render()

        action = agent.decide(state)  # softmax概率选择action
        next_state, reward, done, _ = env.step(action)
        agent.store_transition(state, action, reward)  # 新函数 存取这个transition
        state = next_state
        STEP += 1
        episode_reward += reward  # 计算一个回合的奖励

        if done or STEP==step:
            # print("stick for ",step, " steps")
            if train == True:#如果训练则更新网络
                agent.learn()  # 更新策略网络
            break
    return episode_reward


if __name__ == '__main__':
    seed = 456  # 随机种子数
    env = gym.make('CartPole-v0')
    env.seed(seed)
    torch.manual_seed(seed)  # 策略梯度算法方差很大，设置seed以保证复现性
    frist_train = True
    agent = PGAgent(env)
    episode = 500
    TEST = 20  # 测试的轮数
    episode_rewards = []
    if frist_train == True:
        path = 'D:/深度强化学习/code/policy_based/checkpoint/model_params'
        agent.policy_net.load_state_dict(torch.load(path))
        print("预训练参数加载成功")


    for e in range(episode):
        print("-----------------{} Episode--------------------".format(e))

        playPG(env, agent, step =300, train =True, render=False)#学习

        if e % 50 == 0:
            total_reward = 0
            for i in range(TEST):
                e_reward = playPG(env, agent, step=300, train=False, render=True)
                total_reward += e_reward
            ave_reward = total_reward / TEST
            print('e: ', episode, 'Evaluation Average Reward:', ave_reward)
            episode_rewards.append(ave_reward)

        if e% 100 == 0:
            torch.save(agent.policy_net.state_dict(), os.path.join('checkpoint', "model_params"))


    env.close()
    print('------------------done------------------')
    plt.plot(episode_rewards)
    plt.show()