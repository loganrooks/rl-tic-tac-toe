from __future__ import print_function
from collections import defaultdict
from itertools import count
import numpy as np
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions
from torch.autograd import Variable
import torch.nn.functional as f
import pickle

class Environment(object):
    """
    The Tic-Tac-Toe Environment
    """
    # possible ways to win
    win_set = frozenset([(0,1,2), (3,4,5), (6,7,8), # horizontal
                         (0,3,6), (1,4,7), (2,5,8), # vertical
                         (0,4,8), (2,4,6)])         # diagonal
    # statuses
    STATUS_VALID_MOVE = 'valid'
    STATUS_INVALID_MOVE = 'inv'
    STATUS_WIN = 'win'
    STATUS_TIE = 'tie'
    STATUS_LOSE = 'lose'
    STATUS_DONE = 'done'

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset the game to an empty board."""
        self.grid = np.array([0] * 9) # grid
        self.turn = 1                 # whose turn it is
        self.done = False             # whether game is done
        return self.grid

    def render(self):
        """Print what is on the board."""
        map = {0:'.', 1:'x', 2:'o'} # grid label vs how to plot
        print(''.join(map[i] for i in self.grid[0:3]))
        print(''.join(map[i] for i in self.grid[3:6]))
        print(''.join(map[i] for i in self.grid[6:9]))
        print('====')

    def check_win(self):
        """Check if someone has won the game."""
        for pos in self.win_set:
            s = set([self.grid[p] for p in pos])
            if len(s) == 1 and (0 not in s):
                return True
        return False

    def step(self, action):
        """Mark a point on position action."""
        assert type(action) == int and action >= 0 and action < 9
        # done = already finished the game
        if self.done:
            return self.grid, self.STATUS_DONE, self.done
        # action already have something on it
        if self.grid[action] != 0:
            return self.grid, self.STATUS_INVALID_MOVE, self.done
        # play move
        self.grid[action] = self.turn
        if self.turn == 1:
            self.turn = 2
        else:
            self.turn = 1
        # check win
        if self.check_win():
            self.done = True
            return self.grid, self.STATUS_WIN, self.done
        # check tie
        if all([p != 0 for p in self.grid]):
            self.done = True
            return self.grid, self.STATUS_TIE, self.done
        return self.grid, self.STATUS_VALID_MOVE, self.done

    def random_step(self):
        """Choose a random, unoccupied move on the board to play."""
        pos = [i for i in range(9) if self.grid[i] == 0]
        move = random.choice(pos)
        return self.step(move)

    def play_against_random(self, action, render=False):
        """Play a move, and then have a random agent play the next move."""
        state, status, done = self.step(action)
        if render:
            self.render()
        if not done and self.turn == 2:
            state, s2, done = self.random_step()
            if render:
                self.render()
            if done:
                if s2 == self.STATUS_WIN:
                    status = self.STATUS_LOSE
                elif s2 == self.STATUS_TIE:
                    status = self.STATUS_TIE
                else:
                    raise ValueError("???")
        return state, status, done


class Swish(nn.Module):
    def __init__(self, beta=1.0, trainable=False):
        super(Swish, self).__init__()
        self.beta = Variable(torch.FloatTensor([beta]), requires_grad=trainable)

    def forward(self, x):
        return x * f.sigmoid(self.beta * x)


class Policy(nn.Module):
    """
    The Tic-Tac-Toe Policy
    """
    def __init__(self, input_size=27, hidden_size=64, output_size=9):
        super(Policy, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            Swish()
        )
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        fc1_out = self.fc1(x)
        return self.fc2(fc1_out)


def select_action(policy, state):
    """Samples an action from the policy at the state."""
    state = torch.from_numpy(state).long().unsqueeze(0)
    state = torch.zeros(3,9).scatter_(0,state,1).view(1,27)
    logits = policy(Variable(state))
    pr = f.softmax(logits, logits.size(0))
    m = torch.distributions.Categorical(pr)
    action = m.sample()
    log_prob = torch.sum(m.log_prob(action))
    return action.data[0], log_prob

def compute_returns(rewards, gamma=1.0):
    """
    Compute returns for each time step, given the rewards
      @param rewards: list of floats, where rewards[t] is the reward
                      obtained at time step t
      @param gamma: the discount factor
      @returns list of floats representing the episode's returns
          G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + ... 

    >>> compute_returns([0,0,0,1], 1.0)
    [1.0, 1.0, 1.0, 1.0]
    >>> compute_returns([0,0,0,1], 0.9)
    [0.7290000000000001, 0.81, 0.9, 1.0]
    >>> compute_returns([0,-0.5,5,0.5,-10], 0.9)
    [-2.5965000000000003, -2.8850000000000002, -2.6500000000000004, -8.5, -10.0]
    """
    returns = [rewards[-1]]
    for i, reward in enumerate(rewards[-2::-1]):
        return_ = gamma * returns[i] + reward
        returns.append(return_)
    return returns[::-1]

def finish_episode(saved_rewards, saved_logprobs, gamma=1.0):
    """Samples an action from the policy at the state."""
    policy_loss = []
    returns = compute_returns(saved_rewards, gamma)
    returns = torch.Tensor(returns)
    # subtract mean and std for faster training
    returns = (returns - returns.mean()) / (returns.std() +
                                            np.finfo(np.float32).eps)
    for log_prob, reward in zip(saved_logprobs, returns):
        policy_loss.append(-log_prob * reward)
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward(retain_graph=True)
    # note: retain_graph=True allows for multiple calls to .backward()
    # in a single step
    return policy_loss.data[0]

def get_reward(status):
    """Returns a numeric given an environment status."""
    return {
            Environment.STATUS_VALID_MOVE  : 0, # TODO
            Environment.STATUS_INVALID_MOVE: -100,
            Environment.STATUS_WIN         : 10,
            Environment.STATUS_TIE         : 1,
            Environment.STATUS_LOSE        : -10
    }[status]

def train(policy, env, gamma=1.0, n_episodes=50000, log_interval=500):
    """Train policy gradient."""
    optimizer = optim.Adam(policy.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=10000, gamma=0.9)
    running_reward = 0
    episode_invalid_moves = []
    episode_losses = []
    average_returns = []
    win_loss_tie_ratios = []

    for i_episode in range(1,n_episodes+1):
        invalid_moves = 0
        saved_rewards = []
        saved_logprobs = []
        state = env.reset()
        done = False
        while not done:
            action, logprob = select_action(policy, state)
            state, status, done = env.play_against_random(action)
            invalid_moves += 1 if status == Environment.STATUS_INVALID_MOVE else 0
            reward = get_reward(status)
            saved_logprobs.append(logprob)
            saved_rewards.append(reward)

        R = compute_returns(saved_rewards)[0]
        running_reward += R

        loss = finish_episode(saved_rewards, saved_logprobs, gamma)
        episode_losses.append(loss)
        episode_invalid_moves.append(invalid_moves)

        if i_episode % log_interval == 0:
            average_return = running_reward / log_interval
            print('Episode {}\tAverage return: {:.2f}'.format(
                i_episode,
                average_return))
            average_returns.append(average_return)
            win_loss_tie_ratio = play_games(policy, env, n_games=500)
            win_loss_tie_ratios.append(win_loss_tie_ratio)
            running_reward = 0

        if i_episode % (log_interval) == 0:
            torch.save(policy.state_dict(),
                       "ttt/policy-%d.pkl" % i_episode)

        if i_episode % 1 == 0: # batch_size
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
    return policy, win_loss_tie_ratios, episode_invalid_moves, episode_losses, average_returns


def first_move_distr(policy, env):
    """Display the distribution of first moves."""
    state = env.reset()
    state = torch.from_numpy(state).long().unsqueeze(0)
    state = torch.zeros(3,9).scatter_(0,state,1).view(1,27)
    pr = policy(Variable(state))
    return pr.data


def load_weights(policy, episode):
    """Load saved weights"""
    weights = torch.load("ttt/policy-%d.pkl" % episode)
    policy.load_state_dict(weights)

def play_games(policy, env, n_games, render=False):
    results = {Environment.STATUS_WIN: 0,
               Environment.STATUS_LOSE: 0,
               Environment.STATUS_TIE: 0}
    for game in range(n_games):
        state = env.reset()
        done = False
        while not done:
            action, _ = select_action(policy, state)
            state, status, done = env.play_against_random(action, render)
        results[status] += 1
    return results

if __name__ == '__main__':
    import sys
    np.random.seed(411)

    policy = Policy()
    env = Environment()

    if len(sys.argv) == 1:

        # `python tictactoe.py` to train the agent
        hidden_units_list = [32]
        for hidden_units in hidden_units_list:
            policy = Policy(hidden_size=hidden_units)
            policy, ratios, invalid_moves, episode_losses, average_returns = train(policy, env)
            hidden_units_results = {"ratio": ratios,
                                    "invalid": invalid_moves,
                                    "loss": episode_losses,
                                    "return": average_returns}
            with open('ttt/hidden_units_single_{}.pkl'.format(hidden_units), 'wb') as csv_file:
                pickle.dump(hidden_units_results, csv_file, protocol=pickle.HIGHEST_PROTOCOL)
            print(play_games(policy, env, n_games=5, render=True))
            print(play_games(policy, env, n_games=100))
    else:
        ep = int(sys.argv[1])
        load_weights(policy, ep)
        print(first_move_distr(policy, env))
