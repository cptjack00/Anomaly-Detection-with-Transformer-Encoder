import torch
import torch
import numpy as np
from model import make_model
from utils import NoamOpt, loss_backprop

def window_mask(size, config):
    """Mask out the defined positions."""
    attn_shape = (1, size, size)
    window_mask = np.ones(attn_shape).astype('uint8')
    window_mask[:, :, config['pre_mask']:config['post_mask']] = 0 
    return torch.from_numpy(window_mask) == 0