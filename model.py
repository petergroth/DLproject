"""
PyTorch/Pyro Implementation of Recurrent Mixture Density Networks

"""

# Imports
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



# Recurrent Mixture Density network

class FeatureExtractor(nn.Module):
    """
    Class to extract features from the gridded observations.
    """
    
    def __init__(self, input_dim, hidden_dim, LSTM_input):
        super(FeatureExtractor, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim  = input_dim*input_dim
        self.input_to_hidden  = nn.Linear(in_features=self.input_dim, out_features=hidden_dim)
        self.hidden_to_hidden = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.hidden_to_output = nn.Linear(in_features=hidden_dim, out_features=LSTM_input)
        
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(p=0.5)
        
        
    def forward(self, U):
        # Extract features from U
        y = U.view(-1, self.input_dim)
        y = self.dropout(self.elu(self.input_to_hidden(y)))
        y = self.dropout(self.elu(self.hidden_to_hidden(y)))
        y = self.hidden_to_output(y)
        
        return y
    
class ConvFeatureExtractor(nn.Module):
    """
    Class to extract features from the gridded observations.
    """
    
    def __init__(self, input_dim, hidden_dim, LSTM_input, num_filters):
        super(ConvFeatureExtractor, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim  = input_dim
        self.num_filters= num_filters
        self.conv_dim   = int(input_dim * input_dim * num_filters /4) # Divide by four due to 2 by 2 maxpooling
        
        self.conv_1 = nn.Conv2d(in_channels=1,
                                out_channels=num_filters,
                                kernel_size=5,
                                stride=1,
                                padding=2)
        self.conv_to_hidden = nn.Linear(in_features=self.conv_dim, out_features=hidden_dim)
        self.hidden_to_hidden = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.hidden_to_output = nn.Linear(in_features=hidden_dim, out_features=LSTM_input)
         #self.input_to_hidden  = nn.Linear(in_features=self.input_dim, out_features=hidden_dim)
        #self.hidden_to_hidden = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        #self.hidden_to_output = nn.Linear(in_features=hidden_dim, out_features=LSTM_input)
        
        self.elu = nn.ELU()
        self.dropout  = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout2d(p=0.5)
        self.maxpool  = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, U):
        # Extract features from U
        # Shape images
        y = U.view(-1, 1, self.input_dim, self.input_dim)
        # Convolutional layer
        y = self.dropout2(self.maxpool(self.elu(self.conv_1(y))))
        # Dense layers
        y = y.view(-1, self.conv_dim)
        y = self.dropout(self.elu(self.conv_to_hidden(y)))
        y = self.dropout(self.elu(self.hidden_to_hidden(y)))
        # Output
        y = self.hidden_to_output(y)
        
        return y       
        
    
class MDN(nn.Module):
    '''
    Mixture density network. Takes as input the hidden state from an LSTM cell and uses it to extract
    the parameters for a Gaussian Mixture Model.
    '''   
    def __init__(self, LSTM_input, hidden_dim, K, output_dim):
        super(MDN, self).__init__()
        
        # Define parameters
        self.K = K
        self.LSTM_input = LSTM_input
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.tril_indices = torch.tril_indices(row=output_dim, col=output_dim, offset=-1)

        # Dense layers
        self.input_to_hidden  = nn.Linear(in_features=LSTM_input, out_features=hidden_dim)
        self.hidden_to_hidden = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        # Take output of fully connected layer and feed to layers for GMM components
        self.hidden_to_loc      = nn.Linear(in_features=hidden_dim, out_features=K*output_dim)
        self.hidden_to_sigma    = nn.Linear(in_features=hidden_dim, out_features=K*output_dim)
        self.hidden_to_off_diag = nn.Linear(in_features=hidden_dim, out_features=K)
        self.hidden_to_mix      = nn.Linear(in_features=hidden_dim, out_features=K)
        
        # Functions
        self.elu  = nn.ELU()
        self.softmax = nn.Softmax(dim=2)
        self.softplus= nn.Softplus()
        
        # Dropout
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.3)
        
        
    def forward(self, hidden):     
        # Dense layers   
        y = self.dropout1(self.elu(self.input_to_hidden(hidden)))
        y = self.dropout2(self.elu(self.hidden_to_hidden(y)))
        # Compute mean values
        loc   = self.hidden_to_loc(y).view(-1, self.K, self.output_dim)
        # Compute variances (must be positive)
        sigma = self.softplus(self.hidden_to_sigma(y)).view(-1, self.K, self.output_dim)
        # Compute covariances
        cov   = self.hidden_to_off_diag(y).view(-1, self.K, 1)
        # Compute mixture components (must sum to 1)
        pi    = self.softmax(self.hidden_to_mix(y))
        # Create lower triangular matrix such at Sigma = LL^T
        cov_lower = torch.zeros((hidden.shape[0], self.K, self.output_dim, self.output_dim))
        # Insert covariances
        for i in range(self.K):
            # Fill diagonal
            cov_lower[:, i] += torch.diag_embed(sigma[:, i, :])   
            # Fill lower value
            cov_lower[:, i, 1, 0] = cov[:, i, :]
            
        return loc, pi, cov_lower 

    
class RMDN(nn.Module):
    def __init__(self, input_dim, hidden_dim, LSTM_input, output_dim, K):
        super(RMDN, self).__init__()
        # Initialise feature extractor and MDN
        #self.FeatureExtractor = FeatureExtractor(input_dim, hidden_dim, LSTM_input)
        self.FeatureExtractor = ConvFeatureExtractor(input_dim, hidden_dim, LSTM_input, 10)
        self.MDN              = MDN(LSTM_input, hidden_dim, K, output_dim)
        # Define LSTM layer
        self.LSTM  = nn.LSTM(input_size=LSTM_input, hidden_size=hidden_dim, num_layers=1)

        # Define parameters
        self.input_dim = input_dim*input_dim
        self.hidden_dim= hidden_dim
        self.LSTM_input= LSTM_input
        self.output_dim= output_dim
        self.K = K
        
        self.hidden_h = nn.Parameter(torch.zeros(LSTM_input))
        self.hidden_c = nn.Parameter(torch.zeros(LSTM_input))
        
    @config_enumerate
    def model(self, X=None, U=None, mask=None, batch_size=1):
               
        # Extract batch information
        N = len(U)
        T_max = U.size(1)
        b = min(N, batch_size)
        
        # Allocation of samples and GMM parameters
        x_samples = torch.zeros((b, T_max, max(mask), 2))
        mix = torch.zeros((b, T_max, self.K))
        covs= torch.zeros((b, T_max, self.K, 2, 2))
        locs= torch.zeros((b, T_max, self.K, 2))
    
        # Register module with Pyro 
        pyro.module("RMDN", self)
    
        # Initialise and reshape hidden states 
        hidden = (self.hidden_h.view(b, 1, self.hidden_dim), self.hidden_c.view(b, 1, self.hidden_dim))
        
        # Main plate
        with pyro.plate("data", N, dim=-2):
            # Iterate through each time interval
            for t in pyro.markov(range(0, T_max)):
                # Extract features from U_t
                U_features = self.FeatureExtractor(U[:, t, :, :])
                # Feed features through LSTM
                _, hidden  = self.LSTM(U_features.view(b, 1, self.LSTM_input), hidden)
                # Extract GMM parameters from hidden state
                loc, pi, cov = self.MDN(hidden[0])
                # Save distribution parameters
                locs[:, t, :, :] = loc
                mix[:, t, :] = pi
                covs[:, t, :, :, :] = cov
                
                 # Enter plate for each observation in time step
                with pyro.plate('density_%d'%t, size=mask[t], dim=-1):
                    # Draw which component
                    assignment = pyro.sample(f'assignment_{t}', dist.Categorical(pi.view(b, 1, -1)))                          
                    # Create distributions
                    fn_dist = MN(loc=Vindex(loc)[..., assignment, :], scale_tril=Vindex(cov)[..., assignment, :, :])            
                    # Draw samples
                    if X is None:
                        x_samples[:, t, :mask[t]]    = pyro.sample('x_%d'%t, fn_dist, obs=None)                      
                    else:
                        x_samples[:, t, :mask[t], :] = pyro.sample('x_%d'%t, fn_dist, obs=X[:, t, :mask[t], :])
        
            # Return samples and distribution parameters
            return x_samples, locs, mix, covs
    
    def guide(self, X=None, U=None, mask=None, hidden=None):
        pass

                   
    def get_loglikelihood(self, X, U, mask, X_init, U_init, mask_init):
        with torch.no_grad():
           
            total_num = mask.sum()
            loglik = torch.zeros((total_num))
            counter = 0
            T_max = U.size(1)
            
            # If no initial U image, compute training LL
            if U_init is None:
                # Feed forward
                _, locs, mix, covs = self.model(X=X, U=U, mask=mask)
                
                # For each time interval
                for t in range(0, T_max):
                    # Distributions for current time interval
                    fn_dist = MN(loc=locs[:,t, :, :], scale_tril=covs[:, t, :, :, :])
                    # Compute LL for all data points                    
                    for tt in range(0, mask[t]):
                        loglik[counter] = torch.log((mix[:, t, :]*torch.exp(fn_dist.log_prob(X[:, t, tt, :].squeeze()))).sum()+1e-16)
                        counter += 1
            else:
                # Concatenate
                U_cat = torch.cat((U_init, U), 1)
                X_cat = torch.cat((X_init, X), 1)
                mask_cat = np.hstack((mask_init, mask))
                # Feed all data through network
                _, locs, mix, covs = self.model(X=X_cat, U=U_cat, mask=mask_cat)
                # Discard initialisation parameters
                T_init = U_init.size(1)
                locs = locs[:, T_init:, :, :]
                mix  = mix[:,  T_init:, :]
                covs = covs[:, T_init:, :, :, :]
                
                # For each time interval
                for t in range(0, T_max):
                    # Distributions for current time interval
                    fn_dist = MN(loc=locs[:,t, :, :], scale_tril=covs[:, t, :, :, :])
                    # Compute LL for all data points
                    for tt in range(0, mask[t]):
                        loglik[counter] = torch.log((mix[:, t, :]*torch.exp(fn_dist.log_prob(X[:, t, tt, :].squeeze()))).sum()+1e-16)
                        counter += 1
                           
        return loglik