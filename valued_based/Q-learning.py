import gym
import numpy as np
import matplotlib.pyplot as plt

#用Q-learning解决出租车问题

class QlearningAgent:
    def __init__(self,env, gamma=0.9, learning_rate=0.1, epsilon=0.1):
        self.gamma = gamma
        self.learing_rate = learning_rate
        self.epsilon = epsilon
        self.action_n = env.action_space.n
        self.q = np.zeros((env.observation_space.n, env.action_space.n))  # 初始化Q表
    def decide(self,state):
        if np.random.uniform()>self.epsilon:
            action = self.q[state].argmax()
        else:
            action = np.random.randint(self.action_n)
        return action
    def learn(self, state, action, reward, next_state,done):
        u = reward + self.gamma * self.q[next_state].max() * (1. - done)
        td_error = u - self.q[state, action]
        self.q[state, action] += self.learing_rate * td_error

def play_qlearning(env, agent ,render=False, train = False):
    #print('---------------------下一个回合开始-------------------------')
    ktl = {0:'R',1:'G',2:'Y',3:'B',4:'on'}
    kta = {0:'South',1:'North',2:'East',3:'West',4:'Pickup',5:'Dropoff'}
    episode_reward = 0 #回合总奖励
    observation = env.reset() #开始回合

    #print(len(observation))
    while True:
        taxirow, taxicol, passloc, destidx = env.unwrapped.decode(observation)  # 从500个状态中解码
        if render:
            env.render()
            print('出租车位置 = {}'.format((taxirow, taxicol)))
            print('乘客位置 = '+ ktl[passloc])
            print('目的地位置 = '+ ktl[destidx])
           # print('动作是: '+ kta[action])

        action = agent.decide(observation)  # 从智能体中拿到动作
        next_observation , reward , done, _ = env.step(action)#执行动作，得到奖励，与下一个时刻的状态

        episode_reward += reward  #计算一个回合的奖励
        if train:#是否训练策略/价值网络
            agent.learn(observation, action, reward, next_observation , done)
        if done:
            break
        observation =next_observation

    return episode_reward

if __name__ == '__main__':
    env = gym.make('Taxi-v3')
    agent = QlearningAgent(env)
    episode=500
    episode_rewards= []
    for episode_reward in range(episode):
        episode_reward = play_qlearning(env, agent, render= True,train= True)
        episode_rewards.append(episode_reward)

    plt.plot( episode_rewards)
    plt.show()