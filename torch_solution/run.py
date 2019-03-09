# run.py
# coding: utf-8
import argparse
import os
import numpy as np
from unityagents import UnityEnvironment
from default import get_cfg_defaults

parser = argparse.ArgumentParser(description="Banana Game Collector Setup")
parser.add_argument(
    "--config-file",
    default=None,
    metavar="FILE",
    help="path to config file",
    type=str,
    )
parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
args = parser.parse_args()

"""
Setup all required settings
"""
cfg = get_cfg_defaults()
if args.config_file is not None:
    cfg.merge_from_file(args.config_file)
if args.opts is not None:
    cfg.merge_from_list(args.opts)
cfg.freeze()
env = UnityEnvironment(file_name=cfg.SYSTEM.GAME_LOC)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
if cfg.SYSTEM.DEVICE is not "":
    print("Use GPU")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.SYSTEM.DEVICE)
else:
    print("Use CPU")


# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

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

"""
Test the Game with random player
"""
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

