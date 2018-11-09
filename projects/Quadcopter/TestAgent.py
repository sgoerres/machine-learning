
## TODO: Train your agent here.
import csv
import sys
import pandas as pd
import numpy as np
# from agents.policy_search import PolicySearch_Agent
from agents.Agent import DDPG
from MyTakeOffTask import MyTakeOffTask

num_episodes = 1000
target_pos = np.array([0., 0., 20.])
init_pose = np.array([0., 0., 0.1, 0., 0., 0.])
task = MyTakeOffTask(init_pose=init_pose, target_pos=target_pos)
agent = DDPG(task)

file_output = 'myData.csv'
labels = ['episode', 'time', 'x', 'y', 'z', 'phi', 'theta', 'psi', 'x_velocity',
          'y_velocity', 'z_velocity', 'phi_velocity', 'theta_velocity',
          'psi_velocity', 'rotor_speed1', 'rotor_speed2', 'rotor_speed3', 'rotor_speed4', 'reward']
results = {x: [] for x in labels}

with open(file_output, 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(labels)
    for i_episode in range(1, num_episodes + 1):

        state = agent.reset_episode()  # start a new episode
        while True:
            action = agent.act(state)
            next_state, reward, done = task.step(action)
            agent.step(action, reward, next_state, done)
            state = next_state
            to_write = [i_episode] + [task.sim.time] + list(task.sim.pose) + list(task.sim.v) + list(
                task.sim.angular_v) + list(action) + [reward]
            for ii in range(len(labels)):
                results[labels[ii]].append(to_write[ii])

            writer.writerow(to_write)

            if done:
                print("\rEpisode = {:4d}".format(
                    i_episode), end="")  # [debug]
                break

        sys.stdout.flush()