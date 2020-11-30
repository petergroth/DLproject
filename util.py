"""
Utility functions for Recurrent Mixture Density Network
"""

import pandas as pd
import numpy as np
import torch
from torch import nn
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, TraceEnum_ELBO, config_enumerate
from pyro.optim import Adam, ClippedAdam
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pyro.distributions import MultivariateNormal as MN
from pyro.ops.indexing import Vindex




class RFNDataset(Dataset):
    """Spatio-temporal demand modelling dataset."""
    def __init__(self, X, U):
        self.X = X
        self.U = U

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        X_i, U_i = self.X[idx].float(), self.U[idx].float()
        return X_i, U_i
    
    
#def visualise_predictions()