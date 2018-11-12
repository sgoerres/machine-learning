import numpy as np
from physics_sim import PhysicsSim
import math

class MyTakeOffTask():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 1

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.init_pose = init_pose
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):
        # get all necessary distances
        distance_length_x_max = abs(self.init_pose[0] - self.target_pos[0])
        distance_length_y_max = abs(self.init_pose[1] - self.target_pos[1])
        distance_length_z_max = abs(self.init_pose[2] - self.target_pos[2])

        distance_vectormax = self.init_pose[:3] - self.target_pos
        distance_vector_squared_max = np.square(distance_vectormax)
        distance_length_max = math.sqrt(np.sum(distance_vector_squared_max))
        distance_vector = self.target_pos-self.sim.pose[:3]
        distance_vector_squared = np.square(distance_vector)
        distance_length = math.sqrt(np.sum(distance_vector_squared))
#
        distance_vector2 = self.sim.pose[:3] - self.init_pose[:3]

        distance_vector_squared2 = np.square(distance_vector2)

        distance_length2 = math.sqrt(np.sum(distance_vector_squared2))

        distance_reward_total = 1 - 1/math.sqrt(distance_length / distance_length_max)

        distance_reward_total2 = (distance_length2 / distance_length_max)

        reward = distance_reward_total2 + distance_reward_total

        # punish falling
        if self.sim.pose[2] < self.init_pose[2] and abs(self.sim.pose[2] - self.init_pose[2]) > 0.5:
            reward -= (self.init_pose[2] - self.sim.pose[2]) * 10
#
        # encourage rising
        if self.sim.pose[2] > self.init_pose[2] and abs(self.sim.pose[2] - self.init_pose[2]) > 0.05:
            intermediate = 10 * abs(self.init_pose[2] - self.sim.pose[2])
            reward +=  intermediate #if intermediate <= 2 else 2.0
        # punish velocity in some other direction then up
#
        #if abs(self.sim.v[0]) > 2:
        #    reward -= abs(self.sim.v[0])
#
        #if abs(self.sim.v[1]) > 2:
        #    reward -= abs(self.sim.v[1])
#
        #if abs(self.sim.v[2]) > 2:
        #    reward += abs(self.sim.v[2])
#
        # reward runtime
        reward += self.sim.time #* 2#


        # Step 3: Punish if finished early (taken from https://study-hall.udacity.com/sg-465545-9999/rooms/community:nd009t:465545-cohort-9999-project-1189/community:thread-10335140521-256557?contextType=room)
        if self.sim.done and self.sim.runtime > self.sim.time:
            reward = -100



        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done
    
    def getTotalPathLength(self):
        distance_vector = self.target_pos-self.init_pose[:3]
        distance_vector_squared = np.square(distance_vector)
        distance_length = math.sqrt(np.sum(distance_vector_squared))
        return distance_length
    
    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat) 
        return state