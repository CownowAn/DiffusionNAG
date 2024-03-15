import os
import logging
import torch
import numpy as np
import random

def restore_checkpoint(ckpt_dir, state, device, resume=False):
    if not resume:
        os.makedirs(os.path.dirname(ckpt_dir), exist_ok=True)
        return state
    elif not os.path.exists(ckpt_dir):
        if not os.path.exists(os.path.dirname(ckpt_dir)):
            os.makedirs(os.path.dirname(ckpt_dir))
        logging.warning(f"No checkpoint found at {ckpt_dir}. "
                        f"Returned the same state as input")
        return state
    else:
        loaded_state = torch.load(ckpt_dir, map_location=device)
        for k in state:
            if k in ['optimizer', 'model', 'ema']:
                state[k].load_state_dict(loaded_state[k])
            else:
                state[k] = loaded_state[k]
        return state


def reset_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

