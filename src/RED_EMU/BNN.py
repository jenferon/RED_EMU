import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
import torch.nn as nn
import numpy as np
import pandas as pd
import torch
from pyro.infer import MCMC, NUTS, Predictive
import matplotlib.pyplot as plt

class BNN21(PyroModule):
    def __init__(self, in_dim=3, out_dim=10, hid_dim=[50,100,50], n_hidden_layers=3, prior_std=1.0, prior_mean=0.0):
        """
        Class for a Bayesian Neural Network
        
        Args:
            in_dim: number of astrophysical parameters (e.g., R_bubble, ionising_efficiency, T_vir)
            out_dim: number of k bins in power spectrum (10)
            hid_dim: number of neurons in each hidden layer
            n_hidden_layers: number of hidden layers
            prior_scale: standard deviation of the gaussian prior on weights
        """
        super().__init__()

        self.activation = nn.ELU()  # or nn.ReLU() -- paper uses ELU (exponential linear unit)
        self.n_hidden_layers = n_hidden_layers
        
        self.lay1 = PyroModule[nn.Linear](in_dim, hid_dim[0]) 
        self.lay1.weight = PyroSample(dist.Normal(prior_mean, prior_std).expand([hid_dim[0], in_dim]).to_event(2))
        self.lay1.bias = PyroSample(dist.Normal(0, 1).expand([hid_dim[0]]).to_event(1))

        self.lay2 = PyroModule[nn.Linear](hid_dim[0], hid_dim[1]) 
        self.lay2.weight = PyroSample(dist.Normal(prior_mean, prior_std).expand([hid_dim[1], hid_dim[0]]).to_event(2))
        self.lay2.bias = PyroSample(dist.Normal(0, 1).expand([hid_dim[1]]).to_event(1))
        
        self.lay3 = PyroModule[nn.Linear](hid_dim[1], hid_dim[2]) 
        self.lay3.weight = PyroSample(dist.Normal(prior_mean, prior_std).expand([hid_dim[2], hid_dim[1]]).to_event(2))
        self.lay3.bias = PyroSample(dist.Normal(0, 1).expand([hid_dim[2]]).to_event(1))
        
        self.lay4 = PyroModule[nn.Linear](hid_dim[2], out_dim*2) 
        self.lay4.weight = PyroSample(dist.Normal(prior_mean, prior_std).expand([out_dim*2, hid_dim[2]]).to_event(2))
        self.lay4.bias = PyroSample(dist.Normal(0, 1).expand([out_dim*2]).to_event(1))
        
        self.out_dim = out_dim

    def forward(self, x, y=None):
        """
        Forward pass through the network
        
        Args:
            x: input astrophysical parameters [batch_size, in_dim]
            y: target power spectra [batch_size, out_dim] (optional, for training)
        
        Returns:
            mu: predicted mean power spectrum [batch_size, out_dim]
        """
        # Forward pass
        x = self.activation(self.lay1(x))
        x = self.activation(self.lay2(x))
        x = self.activation(self.lay3(x))
        output = self.lay4(x)
        
        # Split output into mean and log_std
        mu = output[:, :self.out_dim]
        log_std = output[:, self.out_dim:]
        
        # Convert log_std to std (ensure positive)
        sigma = torch.exp(log_std) + 1e-6  # Add small epsilon for numerical stability

        # Sample from the likelihood
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mu, sigma).to_event(1), obs=y)
            
        return mu
        
if __name__ == "__main__":
    #get data
    data = np.load('/home/ppxjf3/repos/RED_EMU/src/RED_EMU/make_data/dataset/training_data.npy') 
    print(data.shape)
    params = pd.read_csv('/home/ppxjf3/repos/RED_EMU/src/RED_EMU/make_data/dataset/training_labels.csv')
    
    params_train = np.zeros([100,3]) #labels need to be in shape
    params_train[:,0] = params['R_bubble'].values
    mean_Rbub = np.mean(params['R_bubble'].values)
    std_Rbub = np.std(params['R_bubble'].values)
    params_train[:,1] = params['Ionising_effciency']
    mean_IonEff = np.mean(params['Ionising_effciency'].values)
    std_IonEff = np.std(params['Ionising_effciency'].values)
    params_train[:,2] = params['T_vir(min)']
    mean_Tvir = np.mean(params['T_vir(min)'].values)
    std_Tvir = np.std(params['T_vir(min)'].values)
    
    prior_means = torch.tensor([mean_Rbub, mean_IonEff, mean_Tvir])
    prior_stds  = torch.tensor([std_Rbub, std_IonEff, std_Tvir])
    # Convert data to PyTorch tensors
    x_train = torch.from_numpy(params_train).float()
    y_train = torch.from_numpy(data[:,0,0]).float()
    
    model = BNN21(out_dim=1,in_dim=x_train.shape[1], prior_mean=mean_Rbub, prior_std=std_Rbub)#prior_scale=priors_train)
    # Set Pyro random seed
    pyro.set_rng_seed(42)
    # Define Hamiltonian Monte Carlo (HMC) kernel
    # NUTS = "No-U-Turn Sampler" (https://arxiv.org/abs/1111.4246), gives HMC an adaptive step size
    nuts_kernel = NUTS(model, jit_compile=False)  # jit_compile=True is faster but requires PyTorch 1.6+

    # Define MCMC sampler, get 50 posterior samples
    mcmc = MCMC(nuts_kernel, num_samples=500,  warmup_steps=500)
    
    # Run MCMC
    mcmc.run(x_train, y_train)
    
    predictive = Predictive(model=model, posterior_samples=mcmc.get_samples())
    x_test = x_train[:3]
    preds = torch.squeeze(torch.Tensor(predictive(x_test)["obs"])) #creates a prediction of shape [num_samples, len(test_data)]
    print(preds.shape)
    """
    mus = np.zeros([len(x_test)])
    stds = np.zeros([len(x_test)])
    for ii in range(0, len(x_test)):
        mus[ii] = np.mean(preds[:,ii].cpu().detach().numpy())
        stds[ii] = np.std(preds[:,ii].cpu().detach().numpy())
        print("truth: {}, predicted mu {} and std {}".format(y_train[ii],mus[ii],stds[ii]))
        
        y_pred = preds['obs'].T.detach().numpy().mean(axis=1)
        y_std = preds['obs'].T.detach().numpy().std(axis=1)

        fig, ax = plt.subplots(figsize=(10, 5))
        plt.xlabel("k", fontsize=30)
        plt.ylabel(r"$ \Delta_2"", fontsize=30)

        ax.plot(x_true, y_true, 'b-', linewidth=3, label="true function")
        ax.plot(x_obs, y_obs, 'ko', markersize=4, label="observations")
        ax.plot(x_obs, y_obs, 'ko', markersize=3)
        ax.plot(x_test, y_pred, '-', linewidth=3, color="#408765", label="predictive mean")
        ax.fill_between(x_test, y_pred - 2 * y_std, y_pred + 2 * y_std, alpha=0.6, color='#86cfac', zorder=5)

        plt.legend(loc=4, fontsize=15, frameon=False)
        plt.show()
        #test 
    
    #print(np.diff(y_train[:3],preds["obs"].values))"""