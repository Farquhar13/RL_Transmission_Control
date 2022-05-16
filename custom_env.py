import gym
import numpy as np

class threshold_env(gym.Env):
    def __init__(self, n_agents, threshold, n_steps=1e4, buffers=None, max_buffers=None,
               buffer_intervals=None, transmit_and_sense=True):
        self.observation_space = gym.spaces.Box(low=np.NINF, high=np.inf, shape=[3*n_agents])
        self.action_space = gym.spaces.Box(low=np.NINF, high=np.inf, shape=[n_agents])
        self.n_agents = n_agents
        self.threshold = threshold
        self.n_steps = n_steps
        self.current_step = 0
        self.buffers = buffers if buffers != None else [1]*n_agents
        self.max_buffers = max_buffers if buffers !=None else [100]*n_agents
        self.buffer_intervals = buffer_intervals if buffer_intervals != None else [1]*n_agents
        self.transmit_and_sense = transmit_and_sense

    def step(self, actions):
        """
        Observation space:
            - Did transmit: {0=no, 1=yes}
            - Transmission successful: {0=no, 1=yes}
            - Interference-sensed: Ratio # of other transmissions / # of agents
            - Buffer: Ratio of buffer / max_buffer, [0, 1]
        """
        # Increment step
        self.current_step += 1

        n_transmissions = actions.count(1)

        # Get reward, observation, and decrement and/or increment buffers
        obs = []
        rewards = []

        
        for i in range(self.n_agents):
            did_transmit = (actions[i] == 1)
            if did_transmit:
                if (1 <= n_transmissions) and (n_transmissions <= self.threshold):
                    transmission_successful = True
                else:
                    transmission_successful =  False

                if (self.transmit_and_sense == True) and (n_transmissions > 0):
                    # This will cause errors if only one agent
                    interference_sensed = (n_transmissions - 1) / (self.n_agents - 1) # How many others transmitted. Normalized
                else:
                    interference_sensed = 0
            else:
                transmission_successful = False
                interference_sensed = n_transmissions / (self.n_agents - 1)

            # Get reward
            if actions[i] == 1:
                if transmission_successful:
                    # Decrement buffer
                    self.buffers[i] -= 1

                    if self.buffers[i] < 0:
                        raise ValueError("Buffers should not be negative")

                    reward = 1 # successful transmission

                else:
                    reward = -1 # unsuccessful transmission
            elif actions[i] == 0:
                reward = 0 # action = 0, no transmission
            else:
                reward = 0

            rewards.append(reward)

            # Increment buffers
            if self.current_step % self.buffer_intervals[i] == 0:
                if self.buffers[i] < self.max_buffers[i]:
                    self.buffers[i] += 1

            # Get observation
            obs.append([int(did_transmit), int(transmission_successful), interference_sensed, self.buffers[i] / self.max_buffers[i]])

        # Get "done"
        if self.current_step == self.n_steps:
            done = True
        else:
            done = False

        # Get info
        info = []

        # print("step", self.current_step)
        # print("action", actions)
        # print("buffer", self.buffers,"\n")

        return obs, rewards, done, info

    def reset(self):
        self.current_step = 0
        self.buffers = [1] * self.n_agents
        starting_obs = [0] * (4 * self.n_agents)
        return starting_obs
