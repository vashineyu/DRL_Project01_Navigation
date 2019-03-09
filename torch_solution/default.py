#defaults.py
"""
Default setting for anything
"""
from yacs.config import CfgNode as CN

_C = CN()

"""
System Setup
"""
_C.SYSTEM = CN()
_C.SYSTEM.GAME_LOC = "../Banana_Linux_NoVis/Banana.x86_64"
_C.SYSTEM.DEVICE = ""
_C.SYSTEM.SAVE_PATH = "./results"

"""
Agent Setup
"""
_C.AGENT = CN()
_C.AGENT.NUM_EPISODE = 2000
_C.AGENT.TRAINING_MODE = True
_C.AGENT.BUFFER_SIZE = int(1e5)
_C.AGENT.UPDATE_EVERY = 25

"""
Model Setup
"""
_C.MODEL = CN()
_C.MODEL.BATCH_SIZE = 512
_C.MODEL.GAMMA = 0.99
_C.MODEL.TAU = 1e-3
_C.MODEL.NEURONS_OF_LAYERS = [64, 64, 64]
_C.MODEL.USE_BN = True

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()