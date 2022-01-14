import torch
import gym
import time
import numpy as np
import torch.nn as nn
import copy
import argparse
import os
from torch.utils.tensorboard import SummaryWriter

#输入是状态S，输出是确定的动作值A
class ActorNet(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound):
        super(ActorNet, self).__init__()
        self.action_bound = torch.tensor(action_bound)

        # layer
        self.layer_1 = nn.Linear(state_dim, 30)
        nn.init.normal_(self.layer_1.weight, 0., 0.3)
        nn.init.constant_(self.layer_1.bias, 0.1)
        # self.layer_1.weight.data.normal_(0.,0.3)
        # self.layer_1.bias.data.fill_(0.1)
        self.output = nn.Linear(30, action_dim)
        self.output.weight.data.normal_(0., 0.3)
        self.output.bias.data.fill_(0.1)

    def forward(self, s):
        a = torch.relu(self.layer_1(s))
        a = torch.tanh(self.output(a))
        # 对action进行放缩，实际上a in [-1,1]
        scaled_a = a * self.action_bound
        return scaled_a


# Critic输入的是当前的state以及Actor输出的action,输出的是Q-value
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        n_layer = 30
        # layer
        self.layer_1 = nn.Linear(state_dim, n_layer)
        nn.init.normal_(self.layer_1.weight, 0., 0.1)
        nn.init.constant_(self.layer_1.bias, 0.1)

        self.layer_2 = nn.Linear(action_dim, n_layer)
        nn.init.normal_(self.layer_2.weight, 0., 0.1)
        nn.init.constant_(self.layer_2.bias, 0.1)

        self.output = nn.Linear(n_layer, 1)

    def forward(self, s, a):
        s = self.layer_1(s)
        a = self.layer_2(a)
        q_val = self.output(torch.relu(s + a))
        return q_val

class Replayer(object):
    def __init__(self,  state_dim, action_dim, batch_size =32, memory_capacticy = 1000):
        super(Replayer, self).__init__()
        self.index = 0  # 索引
        self.count = 0  # 计数
        self.batch_size = batch_size
        self.memory_capacitcy = memory_capacticy  # 容量
        # self.state_dim = state_dim
        # self.action_dim = action_dim
        self.memory  = np.zeros((self.memory_capacitcy, state_dim * 2 + action_dim + 1))

    def store(self,s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.count % self. memory_capacitcy
        self.memory[index, :] = transition
        self.count += 1

    def sample(self):
        indices = np.random.choice(self.memory_capacitcy, size=self.batch_size)
        return self.memory[indices, :]



class DDPGAgent(object):
    def __init__(self, env, replacement, gamma=0.9, lr_a=0.001, lr_c=0.002):
        super(DDPGAgent, self).__init__()
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_bound = env.action_space.high
        self.replacement = replacement
        self.t_replace_counter = 0
        self.gamma = gamma
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.memory =  Replayer(self.state_dim, self.action_dim)

        self.actor = ActorNet(self.state_dim, self.action_dim, self.action_bound)
        self.actor_target = ActorNet(self.state_dim, self.action_dim, self.action_bound)
        # 定义 Critic 网络
        self.critic = Critic(self.state_dim, self.action_dim)
        self.critic_target = Critic(self.state_dim, self.action_dim)
        # 定义优化器
        self.aopt = torch.optim.Adam(self.actor.parameters(), lr=lr_a)
        self.copt = torch.optim.Adam(self.critic.parameters(), lr=lr_c)
        self.mse_loss = nn.MSELoss()

    def decide(self, s):
        s = torch.FloatTensor(s)
        action = self.actor(s)
        return action.detach().numpy()

    def learn(self):
        # soft replacement and hard replacement
        # 用于更新target网络的参数
        if self.replacement['name'] == 'soft':
            # soft的意思是每次learn的时候更新部分参数
            tau = self.replacement['tau']
            a_layers = self.actor_target.named_children()
            c_layers = self.critic_target.named_children()
            for al in a_layers:
                a = self.actor.state_dict()[al[0] + '.weight']
                al[1].weight.data.mul_((1 - tau))
                al[1].weight.data.add_(tau * self.actor.state_dict()[al[0] + '.weight'])
                al[1].bias.data.mul_((1 - tau))
                al[1].bias.data.add_(tau * self.actor.state_dict()[al[0] + '.bias'])
            for cl in c_layers:
                cl[1].weight.data.mul_((1 - tau))
                cl[1].weight.data.add_(tau * self.critic.state_dict()[cl[0] + '.weight'])
                cl[1].bias.data.mul_((1 - tau))
                cl[1].bias.data.add_(tau * self.critic.state_dict()[cl[0] + '.bias'])

        else:
            # hard的意思是每隔一定的步数才更新全部参数
            if self.t_replace_counter % self.replacement['rep_iter'] == 0:
                self.t_replace_counter = 0
                a_layers = self.actor_target.named_children()
                c_layers = self.critic_target.named_children()
                for al in a_layers:
                    al[1].weight.data = self.actor.state_dict()[al[0] + '.weight']
                    al[1].bias.data = self.actor.state_dict()[al[0] + '.bias']
                for cl in c_layers:
                    cl[1].weight.data = self.critic.state_dict()[cl[0] + '.weight']
                    cl[1].bias.data = self.critic.state_dict()[cl[0] + '.bias']

            self.t_replace_counter += 1
            # 从记忆库中采样bacth data
        bm = self.memory.sample()
        bs = torch.FloatTensor(bm[:, :self.state_dim])
        ba = torch.FloatTensor(bm[:, self.state_dim:self.state_dim + self.action_dim])
        br = torch.FloatTensor(bm[:, -self.state_dim - 1: -self.state_dim])
        bs_ = torch.FloatTensor(bm[:, -self.state_dim:])

        # 训练Actor
        a = self.actor(bs)
        q = self.critic(bs, a)
        a_loss = -torch.mean(q)
        self.aopt.zero_grad()
        a_loss.backward(retain_graph=True)
        self.aopt.step()

        # 训练critic
        a_ = self.actor_target(bs_)#没有加噪声
        q_ = self.critic_target(bs_, a_).detach()
        q_target = br + self.gamma * q_
        q_eval = self.critic(bs, ba)
        td_error = self.mse_loss(q_target, q_eval)
        self.copt.zero_grad()
        td_error.backward()
        self.copt.step()

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.copt.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.aopt.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.copt.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.aopt.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)

def Train_DDPG_Once(env, agent, expl_noise, MAX_EP_STEPS = 200, render = False):
    episode_reward = 0  # 回合总奖励
    observation = env.reset()
    done = False
    for j in range(MAX_EP_STEPS):

    # print(len(observation))
        if render:
            env.render()

        action = agent.decide(observation)  # 从智能体中拿到动作
        action = (
                action+ np.random.normal(0, 2 * expl_noise, size = 1)
        ).clip(-2 , 2)  # 在动作选择上添加随机噪声，使动作值限定在某个范围内
        next_observation, reward, done, _ = env.step(action)  # 执行动作，得到奖励，与下一个时刻的状态
        agent.memory.store(observation, action, reward/10, next_observation)
        episode_reward += reward  # 计算一个回合的奖励

        if agent.memory.count > agent.memory.memory_capacitcy:
             #VAR *= .9995#随着训练次数的增加，噪声衰减
             #print("开始学习")
             agent.learn()
        observation = next_observation
        # if j == MAX_EP_STEPS - 1:#因为环境不会自动停止，设置自动break来停止
        #    # print('Episode:', i, ' Reward: %i' % int(episode_reward))
        #     if episode_reward > -300: render = True#当回报满足时，打开渲染
            #break
    return  episode_reward

def eval_polic_once(agent, env, render = False):
    total_reward = 0
    state, done = env.reset(), False

    while not done:
        if render:
            env.render()
        action = agent.decide(state)
        state_next, reward, done, _ = env.step(action)
        total_reward += reward

        state = state_next

    return total_reward

if __name__ == '__main__':
    #设置超参数
    parser = argparse.ArgumentParser()
    #parser.add_argument("--policy", default="TD3")  # Policy name (TD3, DDPG or OurDDPG)
    # parser.add_argument("--env", default="HalfCheetah-v2")          # OpenAI gym environment name
    parser.add_argument("--env", default='Pendulum-v1')
    parser.add_argument("--seed", default=1, type=int)  # Sets Gym, PyTorch and Numpy seeds
    #parser.add_argument("--start_timesteps", default=25e3, type=int)  # Time steps initial random policy is used
    parser.add_argument("--MAX_EP_STEPS", default=200, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--MAX_EPISODES", default=1000, type=int)  # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=32, type=int)  # Batch size for both actor and critic
    parser.add_argument("--gamma", default=0.99)  # Discount factor
    parser.add_argument("--lr_a", default=0.001)  # actor network update rate
    parser.add_argument("--lr_c", default=0.002)  # critic network update rate
    #parser.add_argument("--policy_noise", default=0.2)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)  # Range to clip target policy noise
    #parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true",default=True)  # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")  # Model load file name, "" doesn't load, "default" uses file_name
    args = parser.parse_args()

    #打印环境参数
    file_name = f"{args.env}_{args.seed}"
    print("---------------------------------------")
    print(f"Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    #生成环境
    env = gym.make(args.env)
    env = env.unwrapped

    #放置随机种子
    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed)
    env.seed(1)
    env.action_space.seed(args.seed)

    # 不同的网络更新方式
    REPLACEMENT = [
        dict(name='soft', tau=0.01),
        dict(name='hard', rep_iter=600)
    ][0]

    agent = DDPGAgent(env = env, replacement = REPLACEMENT)

    if args.load_model != "":
        dict_file = file_name if args.load_model == "default" else args.load_model
        agent.load(f"./models/{dict_file}")

    writer = SummaryWriter()
    episode_reward = 0
    t1 = time.time()
    for i in range(args.MAX_EPISODES):
        episode_reward = Train_DDPG_Once(env, agent, expl_noise=args.expl_noise, MAX_EP_STEPS = args.MAX_EP_STEPS, render = False)
        print('Episode:', i, ' Reward: %i' % int(episode_reward))
        writer.add_scalar('episode_reward', episode_reward, i)

    if args.save_model:
       agent.save(f"./models/{file_name}")
    print('Running time: ', time.time() - t1)
    env.close()