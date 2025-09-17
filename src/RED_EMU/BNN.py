import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
import torch.nn as nn


class BNN21(PyroModule):
    def __init__(self, in_dim=3, out_dim=1, hid_dim=5, prior_scale=10.):
        """
        Class for a Bayesian Neural Network
        
        In:
        in_dim: number of parameter to train on (e.g. astro params: ionising effciciency, R_mfp, virial temperature)
        out_dim: number of parameters to output (e.g. number of k bins, 7)
        hid_dim: number of dimensions in hidden layer (we want 3 hidden layers so can make this an array)
        prior_scale: prior data set to help train (WHAT MAKES IT BAYESIAN OVER ANN)
        """
        super().__init__()

        self.activation = nn.Tanh()  # or nn.ReLU() -- paper uses ELU (exponential linear unit)
        self.layer1 = PyroModule[nn.Linear](in_dim, hid_dim)  # Input to hidden layer
        self.layer2 = PyroModule[nn.Linear](hid_dim, out_dim)  # Hidden to output layer

        # Set layer parameters as random variables
        self.layer1.weight = PyroSample(dist.Normal(0., prior_scale).expand([hid_dim, in_dim]).to_event(2))
        self.layer1.bias = PyroSample(dist.Normal(0., prior_scale).expand([hid_dim]).to_event(1))
        self.layer2.weight = PyroSample(dist.Normal(0., prior_scale).expand([out_dim, hid_dim]).to_event(2))
        self.layer2.bias = PyroSample(dist.Normal(0., prior_scale).expand([out_dim]).to_event(1))

    def forward(self, x, y=None):
        x = x.reshape(-1, 1)
        x = self.activation(self.layer1(x))
        mu = self.layer2(x).squeeze()
        sigma = pyro.sample("sigma", dist.Gamma(.5, 1))  # Infer the response noise

        # Sampling model
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mu, sigma * sigma), obs=y)
        return mu