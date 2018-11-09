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
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.init_pose = init_pose
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self):

        # best case (leading to highest reward = 1):
        # velocity in all directions close to 0 => stable at target position : =.3
        # position close to target pos: =.5
        # phi, theta, psi close to 0 (almost fully stable quadcopter): = .2

        # Step 1: Find out how far away I am from the target (only x,y,z relevant)
        distance_vectormax = self.init_pose[:3] - self.target_pos
        distance_vector_squared_max = np.square(distance_vectormax)
        distance_length_max = math.sqrt(np.sum(distance_vector_squared_max))

        distance_vector = self.target_pos-self.sim.pose[:3]
        
        distance_vector_squared = np.square(distance_vector)
        
        distance_length = math.sqrt(np.sum(distance_vector_squared))

        reward = 1 - ((distance_length)/distance_length_max) # normalize against total path

        # Step 1a: put an incentive out for starting vertically and not wobbling much
        reward -= self.sim.pose[5]  # make psi as close to zero as possible => thus start vertically
        reward -= self.sim.pose[4]  # make theta as close to zero as possible => thus start vertically
        reward -= self.sim.pose[3]  # make phi as close to zero as possible => thus start vertically

        ## Step 2: reward rising from the ground
        #distance_vector2 = self.init_pose[:3] - self.sim.pose[:3]
#
        #distance_vector_squared2 = np.square(distance_vector2)
#
        #distance_l#ength2 = math.sqrt(np.sum(distance_vector_squared2))
        #reward += distance_length2

        # Step 2: Punish if going too fast
        if distance_length < 0.2:
            if abs(self.sim.v[0]) >= 0.2:
                reward -= 0.1
            if abs(self.sim.v[1]) >= 0.2:
                reward -= 0.1
            if abs(self.sim.v[2]) >= 0.2:
                reward -= 0.1
            
        # Step 3: Punish if finished early (taken from https://study-hall.udacity.com/sg-465545-9999/rooms/community:nd009t:465545-cohort-9999-project-1189/community:thread-10335140521-256557?contextType=room)
        if self.sim.done and self.sim.runtime > self.sim.time:
            reward = -1

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