# TODO: your agent here!
import numpy as np
from task import Task

class MyAgent():
    def __init__(self, task):
        # Task (environment) information
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high
        self.action_range = self.action_high - self.action_low

        self.w = np.ones(
            shape=(self.state_size, self.action_size),  # weights for simple linear policy: state_space x action_space
                  ) # start producing actions in a decent range

        # Score tracker and learning parameters
        self.best_w = None
        self.best_score = -np.inf
        self.noise_scale = 0.1

        # Episode variables
        self.reset_episode()

    def reset_episode(self):
        self.total_reward = 0.0
        self.count = 0
        state = self.task.reset()
        return state

    def step(self, reward, done):
        # Save experience / reward
        self.total_reward += reward
        self.count += 1

        # Learn, if at end of episode
        if done:
            self.learn()

    def act(self, state):
        # Choose action based on given state and policy
        
        #TODO:
        # Implement some kind of remembering the previous steps
        # First step is always random (basically: increase/decrease speed of rotors by random number)
        # Following Steps:
        # if reward is larger than before increase action even further (increase speed)
        # if reward is smaller than before retry previous action (same speed as before) and increase smaller amount
        # if reward keeps decreasing for the last steps negate previous action
        
        # add a weight to all previous steps: weigh it by increase in reward
        # adjust the weight based on how long ago this step was
        
        action = np.dot(state, self.w)  # simple linear policy
        #print(action)
        return action

    def learn(self):
        # Learn by kind of stochastic policy search
        
        self.score = self.total_reward / float(self.count) if self.count else 0.0
        if self.score > self.best_score:
            self.best_score = self.score
            self.best_w = self.w
            self.noise_scale = max(0.5 * self.noise_scale, 0.01)
        else:
            self.w = self.best_w
            self.noise_scale = min(2.0 * self.noise_scale, 3.2)
        self.w = self.w + self.noise_scale * np.random.normal(size=self.w.shape)  # equal noise in all directions
        