#!/usr/bin/env python
# coding: utf-8

# # Navigation
# 
# ---
# 
# You are welcome to use this coding environment to train your agent for the project.  Follow the instructions below to get started!
# 
# ### 1. Start the Environment
# 
# Run the next code cell to install a few packages.  This line will take a few minutes to run!
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--numbers_of_episode', default = 2000, type = int)
parser.add_argument('--training_mode', default = 1, type = int)
parser.add_argument('--model_name', default = './model_checkpoints/model.ckpt', type = str)
FLAGS = parser.parse_args()

print("RUN IT")
import pip
pip.main(['-q', 'install', './python'])

from unityagents import UnityEnvironment
import numpy as np

# please do not modify the line below
env = UnityEnvironment(file_name="/data/Banana_Linux_NoVis/Banana.x86_64")
# Environments contain **_brains_** which are responsible for deciding the actions of their associated agents. Here we check for the first brain available, and set it as the default brain we will be controlling from Python.

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


# ### 3. Take Random Actions in the Environment
# In the next code cell, you will learn how to use the Python API to control the agent and receive feedback from the environment.
# Note that **in this coding environment, you will not be able to watch the agent while it is training**, and you should set `train_mode=True` to restart the environment.


env_info = env.reset(train_mode=True)[brain_name] # reset the environment
state = env_info.vector_observations[0]            # get the current state
score = 0                                          # initialize the score
while True:
    action = np.random.randint(action_size)        # select an action
    env_info = env.step(action)[brain_name]        # send the action to the environment
    next_state = env_info.vector_observations[0]   # get the next state
    reward = env_info.rewards[0]                   # get the reward
    done = env_info.local_done[0]                  # see if episode has finished
    score += reward                                # update the score
    state = next_state                             # roll over the state to next time step
    if done:                                       # exit loop if episode finished
        break
    
print("Score: {}".format(score))


# When finished, you can close the environment.

# ### 4. It's Your Turn!
# 
# Now it's your turn to train your own agent to solve the environment!  A few **important notes**:
# - When training the environment, set `train_mode=True`, so that the line for resetting the environment looks like the following:
# ```python
# env_info = env.reset(train_mode=True)[brain_name]
# ```
# - To structure your work, you're welcome to work directly in this Jupyter notebook, or you might like to start over with a new file!  You can see the list of files in the workspace by clicking on **_Jupyter_** in the top left corner of the notebook.
# - In this coding environment, you will not be able to watch the agent while it is training.  However, **_after training the agent_**, you can download the saved model weights to watch the agent on your own machine! 

from dqn_agent import Agent
from tqdm import tqdm
from collections import deque
import os

# if in gpu_mode, setup visible gpu device
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import tensorflow as tf
agent = Agent(state_size=state_size, action_size=action_size, seed=0)

def dqn(brain_name,
        n_episodes = 2500, max_t = 1000, eps_start = 1.0, eps_end = 0.01, eps_decay = 0.995, stop_score = 100.0):
    scores = []
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start
    total_steps = deque(maxlen=100)
    
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name] # reset the environment
        state = env_info.vector_observations[0]            # get the current state
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)        # select an action
            env_info = env.step(action)[brain_name]        # send the action to the environment
            next_state = env_info.vector_observations[0]   # get the next state
            reward = env_info.rewards[0]                   # get the reward
            done = env_info.local_done[0]                  # see if episode has finished
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        #
        total_steps.append(t)
        scores_window.append(score)       # save most recent score
        scores.append(score)
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}, mean steps to done: {:.2f}'.format(i_episode, np.mean(scores_window),np.mean(total_steps)))
        if np.mean(scores_window)>= stop_score:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            #torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            agent.Qnetwork.save_model(model_name = './model_checkpoints/bst_model.ckpt')
            break
    agent.Qnetwork.save_model(model_name = './model_checkpoints/model.ckpt')
    return scores

import pandas as pd
if FLAGS.training_mode:
    scores = dqn(brain_name = brain_name, n_episodes = FLAGS.numbers_of_episode)
    result = pd.DataFrame({'score':scores})
    result.to_csv('./result_log/result.csv')

    ## =========== ##
    #Plot it
    import matplotlib.pyplot as plt
    def get_moving_mean(x, window_size):
        moving_mean = []
        for i,value in enumerate(x):
            if i < window_size:
                window = x[:(i+1)]
            elif (i >= window_size) & (i <= (len(x)-window_size)):
                window = x[i:(i+window_size)]
            else:
                window = x[(i-window_size):]
            moving_mean.append(np.mean(window))
        return moving_mean
    moving_avg = get_moving_mean(scores, 100)

    plt.figure(figsize=(8,6))
    plt.plot(range(len(scores)), scores, 'b-')
    plt.plot(range(len(moving_avg)), moving_avg, 'r-')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid()
    plt.savefig("./img/mean_collected_reward.png")

# ### Test The Agent!
agent = Agent(state_size=state_size, action_size=action_size, seed = 0)
agent.Qnetwork.load_model(model_name = FLAGS.model_name)

env_info = env.reset(train_mode=False)[brain_name] # reset the environment
state = env_info.vector_observations[0]            # get the current state
score = 0                                          # initialize the score
current_step = 0
while True:
    action = agent.act(state)       # select an action
    current_step += 1
    env_info = env.step(action)[brain_name]        # send the action to the environment
    next_state = env_info.vector_observations[0]   # get the next state
    reward = env_info.rewards[0]                   # get the reward
    if reward == -1:
        print("Stupid agent!")
    done = env_info.local_done[0]                  # see if episode has finished
    score += reward                                # update the score
    state = next_state                             # roll over the state to next time step
    print('\rCurrent step: %i, current score: %.1f' % (current_step, score), end="")
    if done:                                       # exit loop if episode finished
        break
    
print("")
print("\r Final Score: {}".format(score))

