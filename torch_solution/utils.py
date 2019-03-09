#utils.py
"""
Utils function here
"""
import numpy as np
import random
from collections import deque

class ExperienceReplay():
    def __init__(self, size=int(1e5)):
        self.size = size
        self.reset()
       
    @property
    def length(self):
        return len(self.buffer)
        
    def reset(self):
        self.buffer = deque(maxlen=self.size)
        
    def append(self, observation):
        self.buffer.append(observation)
        
    def draw_sample(self, sample_size):
        buffer_sample = random.choices(self.buffer, k=sample_size)
        states, actions, rewards, next_states = zip(*buffer_sample)
        states = np.array(states)
        actions = np.expand_dims(np.array(actions), 1)
        rewards = np.expand_dims(np.array(rewards), 1)
        next_states = np.array(next_states)
        return states, actions, rewards, next_states