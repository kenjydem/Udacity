import numpy as np
from collections import namedtuple, deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


class REINFORCE(object):
    """Interacts with and learns from the environment."""

    def __init__(self, env, seed=0, num_trajectories = 1, num_episodes= 1000, plot_every = 100, gamma=1.0, alpha=1e-2, fc1_unit=16, fc2_unit=None):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """

        self.env = env
        self.env.seed(seed)

        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n 

        self.num_trajectories = num_trajectories
        self.num_episodes = num_episodes    # Number of episodes
        self.plot_every = plot_every
        self.alpha = alpha                  # Learning rate
        self.gamma = gamma                  # Discount Factor

        #Policy
        self.policy = Policy(self.state_size, self.action_size, seed, fc1_units=fc1_unit, fc2_units=fc2_unit)
        self.optimizer = optim.Adam(self.policy.parameters(), lr = self.alpha) 
        

    def update(self, rewards, saved_log_probs):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            reward (list): recompense dans un scenario 
            gamma (float): discount factor
        """

        
        policy_loss_mean = 0
        for i in range(self.num_trajectories):

            #Classic
            #discounts = [self.gamma**i for i in range(len(rewards[i])+1)]
            #R = sum([a*b for a,b in zip(discounts, rewards[i])])
           
            # Credit Assignement
            running_add = 0.0
            R = np.zeros_like(rewards[i])
            for t in reversed(range(0, len(rewards[i]))):
                running_add = running_add * self.gamma + rewards[i][t]
                R[t] = running_add

            #Reward reduction
            R = (R- np.mean(R))/ (np.std(R)+1.e-5)

            policy_loss = []
            t = 0
            for log_prob in saved_log_probs[i]:
                policy_loss.append(-log_prob * R[t]); t+=1

            tmp = torch.cat(policy_loss).sum()
            policy_loss_mean += tmp.mul(1/self.num_trajectories)

        
        self.optimizer.zero_grad()
        policy_loss_mean.backward()
        self.optimizer.step()                  

    def learn(self, n_episodes=2000, max_t=1000, target_reward=200.):
        """Deep Q-Learning.
    
        Params
        ======
            n_episodes (int): maximum number of training episodes
            max_t (int): maximum number of timesteps per episode
            eps_start (float): starting value of epsilon, for epsilon-greedy action selection
            eps_end (float): minimum value of epsilon
            eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
        """

        scores = []                        # list containing scores from each episode
        scores_window = deque(maxlen=self.plot_every)  # last 100 scores

        for i_episode in range(1, n_episodes+1):

            if (i_episode-1)%self.num_trajectories == 0:
                saved_log_probs = []
                rewards = []
                num_trajectories = 0

            saved_log_probs.append([])
            rewards.append([])

            state = self.env.reset()

            for t in range(max_t):
                action, log_prob = self.policy.act(state)
                saved_log_probs[num_trajectories].append(log_prob)
                state, reward, done, _ = self.env.step(action)
                rewards[num_trajectories].append(reward)

                if done:
                    break 

            # MAJ de la politique
            if (i_episode)%self.num_trajectories == 0:
                self.update(rewards, saved_log_probs)

            # Collect info par episode
            scores_window.append(sum(rewards[num_trajectories]))
            scores.append(sum(rewards[num_trajectories]))

            num_trajectories += 1

            if i_episode % self.plot_every == 0:
                print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))

            if np.mean(scores_window)>=target_reward:
                print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
                break
        
        return scores


class Policy(nn.Module):
    """Policy Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=16, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Policy, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)

        if fc2_units==None:
            self.fc2 = nn.Linear(fc1_units, action_size)
            self.fc3 = None
        else:
            self.fc2 = nn.Linear(fc1_units, fc2_units)
            self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        if self.fc3 == None:
            x = F.softmax(self.fc2(x),dim=1)
        else:
            x = F.relu(self.fc2(x))
            x = F.softmax(self.fc3(x),dim=1)
        return x 

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.forward(state)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

if __name__ == '__main__':
    import gym, random, torch
    import numpy as np
    from collections import deque
    import matplotlib.pyplot as plt
    gym.logger.set_level(40)

    #'CartPole-v1'    ->  n_episodes=5000, max_t=500,target_reward=475
    #'LunarLander-v2' ->  n_episodes=2000, max_t=1000,target_reward=200
    

    #'CartPole-v1' 
    #env = gym.make('CartPole-v1')
    #env.seed(0)
    #agent = REINFORCE(env, fc1_unit=16, num_trajectories = 1)
    #scores = agent.learn(n_episodes=5000, max_t=500,target_reward=495)   #Initial, ca marche bien
    

    #'LunarLander-v2'
    env = gym.make('LunarLander-v2')
    env.seed(0)
    agent = REINFORCE(env, fc1_unit=64, gamma=0.99, num_trajectories = 1) 
    scores = agent.learn(n_episodes=10000, max_t=1000,target_reward=200)

    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

    state = env.reset()
    for t in range(10000):
        state = torch.from_numpy(state).float().unsqueeze(0)
        prob = agent.policy.forward(state)
        action = np.argmax(prob.detach().numpy())
        state, reward, done, _ = env.step(action)
        env.render()
        if done:
            break 

    env.close()
