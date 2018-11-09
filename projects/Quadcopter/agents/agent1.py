# TODO: your agent here!
import numpy as np

class RewardHistory():
    def __init__(self):
        self.lastAction = np.array([0., 0., 0., 0.])
        self.history = list()

    def storeActionReward(self, reward):
        newKey = len(self.history)
        self.history.append((newKey, (self.lastAction, reward)))

    def storeAction(self, action):
        self.lastAction = action

    def getLastActions(self, amount):
        negateAmount = -1 * amount

        # index is in item[0]
        # action is in item[1][0]
        # reward is in item[1][1]

        return list(reversed(self.history))[:amount]

    def getAgeWeightRewards(self, weight=0.1,  amount=100):
        lastActions = self.getLastActions(amount)

        returnValue = list()

        for i in lastActions:
            returnValue.append((i[0], (i[1][0], i[1][1] * weight)))
            weight *= weight

        return returnValue

    #def getActionMaxWeightReward(self, weight=0.1, amount=100):
    def reset(self, maxitems = 50):
        #self.lastAction = np.array([0., 0., 0., 0.])
        #self.history = dict()
        #cleanup really old history
        if len(self.history) > maxitems:
            for i in range(len(self.history) - maxitems):
                self.history.pop(0)

    def getLastActionReward(self):
        items = self.history.items()
        return list(items)[-1][1]

class MyAgent():
    def __init__(self, task):
        # Task (environment) information
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high
        self.action_range = self.action_high - self.action_low

        # Score tracker and learning parameters
        self.best_w = None
        self.best_score = -np.inf
        self.rw = RewardHistory()
        self.maxAge = 50
        self.score = -np.inf
        self.count = 0
        # Episode variables
        self.reset_episode()

    def reset_episode(self):
        self.total_reward = 0.0
        #self.count = 0
        state = self.task.reset()
        self.rw.reset(self.maxAge)
        return state

    def step(self, reward, done):
        # Save experience / reward
        self.rw.storeActionReward(reward)
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

        if self.count <= 1:
            # take a random move around 10% of allowed action space
            randomIncrease = (self.action_range)  * np.random.normal(0.8, 0.1) # scale by anything between 0.1 and 0.9 randomly
            self.rw.lastAction = np.array([randomIncrease, randomIncrease, randomIncrease, randomIncrease]) # all 4 rotors same rpm target
            return self.rw.lastAction
        else:
            # get last rewards
            pastActionsWeight = self.rw.getAgeWeightRewards(0.85, self.maxAge)

            maxReward = -np.inf
            actionForThisReward = np.array([0.,0.,0.,0.])

            for i in pastActionsWeight:
                reward = i[1][1]
                action = i[1][0]
                if reward >= maxReward:
                    maxReward = reward
                    actionForThisReward = action

            randomComponent = np.random.normal(1.0, 0.15) # add some randomness instead of just repeating the action
            return actionForThisReward * randomComponent

    def learn(self):
        # Learn by random policy search, using a reward-based score
        self.score = self.total_reward / float(self.count) if self.count else 0.0
        if self.score > self.best_score:
            self.best_score = self.score
            self.best_w = self.rw.history
        else:
            self.rw.history = self.best_w
