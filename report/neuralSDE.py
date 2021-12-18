import torch.nn as nn
import torch
from torchsde import sdeint
from torchdiffeq import odeint

class NeuralSDE(nn.Module):
    def __init__(self, state_dim, batch_dim):
        super().__init__()
        self.theta = nn.Parameter(torch.tensor(0.1), requires_grad=True)  # Scalar parameter.
        self.noise_type = "diagonal"
        self.sde_type = "ito"
        self.state_dim = state_dim
        self.batch_dim = batch_dim
    
    # Drift function (i.e. f)
    def f(self, t, y):
        return torch.sin(t) + self.theta * y
    
    # Diffusion function (i.e. g)
    def g(self, t, y):
        return 0.3 * torch.sigmoid(torch.cos(t) * torch.exp(-y))
    
    # Dummy function that always returns a tensor with 0
    def zero_drift(self, t, y):
        return torch.zeros(self.batch_dim, self.state_dim)
    
    # Integrate only drift term given the drift function
    # and inital value
    def compute_mean(self, t, y0=None):
        with torch.no_grad():
            if y0 is None:
                y0 = torch.zeros(self.batch_dim, self.state_dim)
            mean = odeint(self.f, y0, t)
        return mean

    # Compute diffusion term give the diffusion function
    # Set drift and intial value to 0
    def compute_noise(self, t):
        with torch.no_grad():
            y0 = torch.zeros(self.batch_dim, self.state_dim)
            noise = sdeint(self, y0, t, method='euler', names={'drift': 'zero_drift'})
        return noise