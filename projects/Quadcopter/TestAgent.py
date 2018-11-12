import matplotlib.pyplot as plt

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
init_pose = np.array([0., 0., 1., 0., 0., 0.])
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

            if done or reward < -20:  # cancel if done or really bad
                print("\rEpisode = {:4d}".format(
                    i_episode), end="")  # [debug]
                break

        sys.stdout.flush()

print("---------")
## TODO: Plot the rewards.
rewardsEp1000 = np.array(results['reward'])[(np.array(results['episode']) == 1000)]
rewardsEp999 = np.array(results['reward'])[(np.array(results['episode']) == 999)]
rewardsEp998 = np.array(results['reward'])[(np.array(results['episode']) == 998)]
rewardsEp997 = np.array(results['reward'])[(np.array(results['episode']) == 997)]
rewardsEp996 = np.array(results['reward'])[(np.array(results['episode']) == 996)]
rewardsEp995 = np.array(results['reward'])[(np.array(results['episode']) == 995)]
rewardsEp994 = np.array(results['reward'])[(np.array(results['episode']) == 994)]
rewardsEp993 = np.array(results['reward'])[(np.array(results['episode']) == 993)]
rewardsEp992 = np.array(results['reward'])[(np.array(results['episode']) == 992)]
rewardsEp991 = np.array(results['reward'])[(np.array(results['episode']) == 991)]
plt.figure()
# plt.scatter(X_train, y_train, c="k", label="training samples")
plt.scatter(range(0, len(rewardsEp1000),1), rewardsEp1000, c="b", label="E1000")
plt.scatter(range(0, len(rewardsEp999),1), rewardsEp999, c="r", label="E999")
plt.scatter(range(0, len(rewardsEp998),1), rewardsEp998, c="y", label="E998")
plt.scatter(range(0, len(rewardsEp997),1), rewardsEp997, c="c", label="E997")
plt.scatter(range(0, len(rewardsEp996),1), rewardsEp996, c="m", label="E996")
plt.scatter(range(0, len(rewardsEp995),1), rewardsEp995, c="g", label="E995")
plt.scatter(range(0, len(rewardsEp994),1), rewardsEp994, c="gray", label="E994")
plt.scatter(range(0, len(rewardsEp993),1), rewardsEp993, c="crimson", label="E993")
plt.scatter(range(0, len(rewardsEp992),1), rewardsEp992, c="slategray", label="E992")
plt.scatter(range(0, len(rewardsEp991),1), rewardsEp991, c="darkkhaki", label="E991")
plt.legend()
plt.show()

## calculate cumulated rewards
sumRewardsEp1000   = np.sum(rewardsEp1000)
sumRewardsEp999    = np.sum(rewardsEp999 )
sumRewardsEp998    = np.sum(rewardsEp998 )
sumRewardsEp997    = np.sum(rewardsEp997 )
sumRewardsEp996    = np.sum(rewardsEp996 )
sumRewardsEp995    = np.sum(rewardsEp995 )
sumRewardsEp994    = np.sum(rewardsEp994 )
sumRewardsEp993    = np.sum(rewardsEp993 )
sumRewardsEp992    = np.sum(rewardsEp992 )
sumRewardsEp991    = np.sum(rewardsEp991 )

meanOfAll = np.mean(np.array([sumRewardsEp1000, sumRewardsEp999 , sumRewardsEp998 ,sumRewardsEp997 ,sumRewardsEp996 ,sumRewardsEp995 ,sumRewardsEp994 ,sumRewardsEp993 ,sumRewardsEp992 ,sumRewardsEp991 ]))

print(f"Mean of last 10 episode rewards: {meanOfAll}")