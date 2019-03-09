#agent.py
"""
Claim Agent (player) here
"""
import os
import random
import numpy as np

class DQN_agent():
    """Game agent
    Args:
      - state_size (int): environment state size
      - action_size (int): available action size
      - model_clas (class): model object (that have been compiled)
      - seed (int): random seed number
    Returns:
      - Training instance
    """
    def __init__(self, state_size, action_size, model_clas, seed=1234):
        self.state_size = state_size
        self.action_size = action_size
        self.model = model_clas
        self.seed = random.seed(seed)
        
    
    
if __name__ == "__main__":
    pass
