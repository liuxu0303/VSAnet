from __future__ import division
import torch

def save_checkpoint(save_path, timestamp, vsanet_state, filename='checkpoint.pth.tar'):
    file_prefixes = ['vsanet']
    states = [vsanet_state]
    for (prefix, state) in zip(file_prefixes, states):
        torch.save(state, save_path/'{}_{}_{}'.format(timestamp, prefix, filename))
