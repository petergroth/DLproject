"""
Utility functions for Recurrent Mixture Density Network
"""

# Misc
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture

# Visualiation tools
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import animation, rc
from IPython.display import HTML, Image
from PIL import Image

# Pyro/PyTorch
import torch
from torch import nn
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, TraceEnum_ELBO, Predictive, NUTS, MCMC, config_enumerate
from pyro.infer.autoguide import AutoDelta, AutoDiagonalNormal, AutoMultivariateNormal
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
   


    
(lonMin, lonMax, latMin, latMax) = (12.46, 12.67, 55.62, 55.74)
extent=(lonMin, lonMax, latMin, latMax)
aspect = (lonMax-lonMin)/(latMax-latMin)
latmean = 55.678004123412244
latstd  = 0.02278162594471914
lonmean = 12.561432568225557
lonstd  = 0.03821015225646347
    
def visualise_predictions(LL, X, mask, t=0, figsize=(10,10), filename=None):
    # Unpack X_tensor
    lon = X[:, t, :mask[t], 0].squeeze().numpy()*lonstd+lonmean
    lat = X[:, t, :mask[t], 1].squeeze().numpy()*latstd+latmean
    
    fig = plt.figure(figsize=figsize)
    
    # Plot Copenhagen 
    cph_img = np.asarray(Image.open('images/cph_2.png').convert("L"))
    plt.imshow(cph_img, extent=extent, aspect=aspect, cmap='gray')
    
    # Extract and show
    image = LL[0, t, :, :].squeeze().numpy().T
    plt.imshow(image, interpolation='bicubic',cmap='seismic', aspect='auto', extent=extent
              ,vmin=-7, vmax=2, alpha=0.7, origin='lower')
    
    # Show true values
    plt.scatter(x=lon,y=lat, color='k', s=30)
    
    # Save figure
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
    else:
        plt.show()
    
    
def visualise_baseline_mix(LL, X, mask, t=0, figsize=(10,10), filename=None):   
    # Unpack X_tensor
    lon = X[:, t, :mask[t], 0].squeeze().numpy()*lonstd+lonmean
    lat = X[:, t, :mask[t], 1].squeeze().numpy()*latstd+latmean

    fig = plt.figure(figsize=figsize)
    # Plot Copenhagen 
    cph_img = np.asarray(Image.open('images/cph_2.png').convert("L"))
    plt.imshow(cph_img, extent=extent, aspect=aspect, cmap='gray')
    plt.imshow(LL, interpolation='bicubic',cmap='seismic', aspect='auto', extent=extent
              ,vmin=-7, vmax=2, alpha=0.7, origin='lower')
    plt.scatter(x=lon,y=lat, color='k', s=30)

    # Save figure
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
    else:
        plt.show()
