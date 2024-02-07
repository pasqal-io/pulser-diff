import torch

from pulser_diff.pulser.math import set_comp_backend

set_comp_backend("torch")

torch.set_default_dtype(torch.float64)
