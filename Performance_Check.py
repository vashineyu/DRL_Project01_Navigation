import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--numbers_of_round', default = 5, type = int)
parser.add_argument('--model_name', default = './model_checkpoints/model.ckpt', type = str)
FLAGS = parser.parse_args()

print("RUN IT")
import pip
pip.main(['-q', 'install', './python'])

from unityagents import UnityEnvironment
import numpy as np
# please do not modify the line below
env = UnityEnvironment(file_name="/data/Banana_Linux_NoVis/Banana.x86_64")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]
# number of agents in the environment
print('Number of agents:', len(env_info.agents))

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space 
state = env_info.vector_observations[0]
print('States look like:', state)
state_size = len(state)
print('States have length:', state_size)

from dqn_agent import Agent
from tqdm import tqdm
from collections import deque
import os
import tensorflow as tf

agent = Agent(state_size=state_size, action_size=action_size, seed=0)
agent.Qnetwork.load_model(model_name = FLAGS.model_name)

for game_number in range(1, FLAGS.numbers_of_round+1):
    print("Game round: %i" % game_number)
    env_info = env.reset(train_mode=False)[brain_name] # reset the environment
    state = env_info.vector_observations[0]            # get the current state
    score = 0                                          # initialize the score
    step_elapsed = 0
    while True:
        action = agent.act(state)       # select an action
        step_elapsed += 1
        env_info = env.step(action)[brain_name]        # send the action to the environment
        next_state = env_info.vector_observations[0]   # get the next state
        reward = env_info.rewards[0]                   # get the reward
        if reward == -1:
            print("Stupid agent!")
        done = env_info.local_done[0]                  # see if episode has finished
        score += reward                                # update the score
        state = next_state                             # roll over the state to next time step
        print('\rCurrent elapsed step: %i, current score: %.1f' % (step_elapsed, score), end="")
        if done:                                       # exit loop if episode finished
            break
    print("")
    print("Final Score: {}".format(score))
    print("=========")