from abc import ABC, abstractmethod

class Agent(ABC):
    @abstractmethod
    def __init__(self, state_space, action_space):
        pass

    @abstractmethod
    def choose_action(self, state):
        """ Return the agent's action for the next time-step """
        pass

    @abstractmethod
    def remember(self, state, action, reward, next_state, done):
        """ Store trajectory in the agent's experience replay memory """
        pass

    @abstractmethod
    def learn(self):
        """ Train the agent """
        pass
