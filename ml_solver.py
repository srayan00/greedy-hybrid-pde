import torch
from neuralop.models import FNO
import models

class MLSolver(torch.nn.Module):
    def __init__(self, dim , in_channels = 1):
        super().__init__()
        self.dim = dim
        self.in_channels = in_channels
    
    def forward(self, input):
        raise NotImplementedError

#class DeepONet(torch.nn.Module):
class DeepONet(MLSolver):
    def __init__(self, N, dim, device, in_channels = 1, boundary = "Periodic", branch_dim = 64, hidden_branch = 128, num_branch_layers = 1, hidden_trunk=128, num_trunk_layers = 1):
        """
        N: Resolution 
        dim: 1D/2D PDE - We assume that it lies between 0 and 1
        branch_dim: dimension of the latent embedding from branch net
        hidden_branch: hidden size of the branch net
        num_branch_layers: number of layers in branch net
        hidden_trunk: hidden size of the trunk net
        num_trunk_layers: number of layers in trunk net
        """
        super().__init__(dim, in_channels)
        self.N = N
        self.branch_dim =branch_dim
        self.hidden_branch = hidden_branch
        self.num_branch_layers = num_branch_layers
        self.hidden_trunk = hidden_trunk
        self.num_trunk_layers = num_trunk_layers
        self.boundary = boundary
        xs = torch.linspace(0, 1, N + 1)[:-1] if boundary == "Periodic" else torch.linspace(0, 1, N)
        if self.dim > 1:
            ys = torch.linspace(0, 1, N + 1)[:-1] if boundary == "Periodic" else torch.linspace(0, 1, N)
            xs, ys = torch.meshgrid(xs, ys, indexing = "ij")
            coords = torch.stack([xs, ys], axis = -1) # shape N \times N \times 2
            self.coords = coords.reshape(-1, 2).to(device) # N^2 \times 2
        else:
            self.coords = xs.reshape(-1, 1).to(device) # N \times 1
        self.input_size = N if self.dim == 1 else N*N
        self.input_size = self.in_channels*self.input_size
        branch_hidden_list = [self.input_size] + [hidden_branch]*num_branch_layers + [branch_dim]
        self.branch_net = models.MLP(branch_hidden_list)
        trunk_hidden_list = [dim] + [hidden_trunk]*num_trunk_layers + [branch_dim]
        self.trunk_net = models.MLP(trunk_hidden_list)
    
    def forward(self, input, coords = None):
        """
        input is of size (B, N, N) or (B, N)
        coords is of size (num_coords, 2) if dim == 2 else (num_coords, 1)
        """
        bs = input.size(0)

        input_flat = input.reshape(-1, self.input_size) # (B, N^2)
        branch_output = self.branch_net(input_flat) # (B, branch_dim)
        if coords:
            trunk_output = self.trunk_net(coords) #num_coords, branch_dim
        else:
            trunk_output = self.trunk_net(self.coords) # (N^2, branch_dim)
        trunk_output = trunk_output.transpose(0, 1)
        out = torch.matmul(branch_output, trunk_output)
        out = out.reshape(-1, self.N, self.N) if self.dim ==2 else out.reshape(-1, self.N)
        # impose dirichlet boundary condition
        if self.boundary == "Dirichlet":
            if self.dim ==1:
                out[:, 0] = 0.0
                out[:, -1] = 0.0
            else:
                out[:, 0, :] = 0.0
                out[:, -1, :] = 0.0
                out[:, :, 0] = 0.0
                out[:, :, -1] = 0.0
        return out

"""
ALL THE CODE BELOW HAS NOT BEEN USED. IT CAN BE IGNORED. I JUST WANTED TO KEEP IT JUST IN CASE.
"""

class DeepONetCNN(MLSolver):
    def __init__(self, N, dim, device, in_channels = 1, boundary = "Periodic", hidden_branch_channels = [16, 32, 64], hidden_trunk = [16, 32, 64], hidden_branch = [64, 40, 40], kernel_size = 3, stride = 2):
        super().__init__(dim, in_channels)
        self.N = N
        self.hidden_branch_channels = hidden_branch_channels
        self.hidden_trunk = hidden_trunk
        self.kernel_size = kernel_size
        self.stride = stride
        self.boundary = boundary
        xs = torch.linspace(0, 1, N + 1)[:-1] if boundary == "Periodic" else torch.linspace(0, 1, N)
        if self.dim > 1:
            ys = torch.linspace(0, 1, N + 1)[:-1] if boundary == "Periodic" else torch.linspace(0, 1, N)
            xs, ys = torch.meshgrid(xs, ys, indexing = "ij")
            coords = torch.stack([xs, ys], axis = -1) # shape N \times N \times 2
            self.coords = coords.reshape(-1, 2).to(device) # N^2 \times 2
        else:
            self.coords = xs.reshape(-1, 1).to(device) # N \times 1
        self.part_branch_net = models.CNNBlock(hidden_branch_channels, kernel_size, stride)
        self.second_branch_net = models.MLP(hidden_branch)
        self.trunk_net = models.MLP(hidden_trunk)
        self.input_size = N if self.dim == 1 else N*N
        self.input_size = self.in_channels*self.input_size
    
    def forward(self, input, coords = None):
        """
        input is of size (B, 2, N, N) or (B, 2, N)
        coords is of size (num_coords, 2) if dim == 2 else (num_coords, 1)
        """
        bs = input.size(0)
        
        # input_flat = input.reshape(-1, self.input_size) # (B, N^2)
        first_branch_output = self.part_branch_net(input) # (B, hidden_branch_channels[-1], N, N) or (B, hidden_branch_channels[-1], N)
        second_branch_output = self.second_branch_net(first_branch_output) # (B, hidden_branch[-1])
        if coords:
            trunk_output = self.trunk_net(coords) #num_coords, branch_dim
        else:
            trunk_output = self.trunk_net(self.coords) # (N^2, branch_dim)
        trunk_output = trunk_output.transpose(0, 1)
        out = torch.matmul(second_branch_output, trunk_output)
        out = out.reshape(-1, self.N, self.N) if self.dim ==2 else out.reshape(-1, self.N)
        # impose dirichlet boundary condition
        if self.boundary == "Dirichlet":
            if self.dim ==1:
                out[:, 0] = 0.0
                out[:, -1] = 0.0
            else:
                out[:, 0, :] = 0.0
                out[:, -1, :] = 0.0
                out[:, :, 0] = 0.0
                out[:, :, -1] = 0.0
        return out

class FNOforPDE(MLSolver):
    def __init__(self, trunc_mode, dim, in_channels=1, hidden_size = 32, num_layers = 2):
        super().__init__(dim, in_channels)
        self.fno = FNO(n_modes = (trunc_mode,)*dim,
                       in_channels = in_channels,
                       out_channels = 1,
                       hidden_channels = hidden_size,
                       num_layers = num_layers)
    
    def forward(self, input):
        """
        input is of size (B, 2, N, N) or (B, 2, N) or (B, 1, N, N) or (B, 1, N)
        """
        out = self.fno(input) # (B, 1, N, N) or (B, 1, N)
        return out.squeeze(1)

class PredictorRejector(torch.nn.Module):
    def __init__(self, predictor, rejector):
        super().__init__()
        self.predictor = predictor
        self.rejector = rejector
    
    def forward(self, input, teacher_forcing = False, tf_steps = 0, tf_prob = 1.0, targets = None, scaling = None):
        raise NotImplementedError



