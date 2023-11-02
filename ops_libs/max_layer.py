import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

import pdb


__all__ = ['max_op']


def max_op(input: torch.Tensor, dim=-1):
    max_score, max_index = input.max(dim=dim, keepdim=False)
    max_index = max_index.to(torch.int32)
    return max_score, max_index