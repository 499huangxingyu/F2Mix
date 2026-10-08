"""Common utilities: seeding, checkpointing, and output directories."""

import logging
import os
import random
from collections import OrderedDict

import numpy as np
import torch


def setup_seed(seed: int):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def strip_module_prefix(state_dict) -> OrderedDict:
    """Remove the ``module.`` prefix added by nn.DataParallel."""
    return OrderedDict(
        (k[len('module.'):] if k.startswith('module.') else k, v)
        for k, v in state_dict.items())


def save_checkpoint(epoch: int, model, optimizer, path: str, extra: dict = None):
    """Save a resumable training checkpoint."""
    state = {
        'epoch': epoch,
        'model_state_dict': (model.module if hasattr(model, 'module') else model).state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    if extra:
        state.update(extra)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, model, optimizer=None, device='cpu') -> int:
    """Load a checkpoint into ``model`` (and optionally ``optimizer``).

    Returns the stored epoch number.
    """
    ckpt = torch.load(path, map_location=device)
    model_sd = strip_module_prefix(ckpt['model_state_dict'])
    target = model.module if hasattr(model, 'module') else model
    target.load_state_dict(model_sd)
    if optimizer is not None and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    return ckpt.get('epoch', 0)


def load_weights(path: str, model, device='cpu'):
    """Load plain model weights (e.g. a best-model .pth file)."""
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and 'model_state_dict' in state:
        state = state['model_state_dict']
    target = model.module if hasattr(model, 'module') else model
    target.load_state_dict(strip_module_prefix(state))
    return model


def make_run_dir(root: str, backbone: str, tag: str = '') -> str:
    """Create and return ``<root>/<backbone>[_<tag>]`` with subfolders."""
    run_dir = os.path.join(root, backbone + (f'_{tag}' if tag else ''))
    for sub in ('checkpoints', 'models', 'test'):
        os.makedirs(os.path.join(run_dir, sub), exist_ok=True)
    return run_dir


def setup_logging(log_file: str):
    """Configure root logging to append to ``log_file`` and echo to stdout."""
    os.makedirs(os.path.dirname(log_file) or '.', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[logging.FileHandler(log_file, mode='a'),
                  logging.StreamHandler()],
        force=True,
    )
    return logging.getLogger('f2mix')
