import torch
import torch.nn as nn

class PyTorchPCENLayer(nn.Module):
    def __init__(self,
                 alpha=0.98,
                 smooth_coef=0.03,
                 delta=2.0,
                 root=0.5,
                 floor=1e-6,
                 trainable=True):
        super(PyTorchPCENLayer, self).__init__()
        
        self.floor = floor
        self.trainable = trainable

        # Trainable parameters
        self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=trainable)
        self.delta = nn.Parameter(torch.tensor(delta), requires_grad=trainable)
        self.root = nn.Parameter(torch.tensor(root), requires_grad=trainable)
        self.smooth_coef = nn.Parameter(torch.tensor(smooth_coef), requires_grad=trainable)
        
        # We also need a smooth_coef_kernel if we want to mirror the Keras way, 
        # but in PyTorch parameter is enough.
        
        # Note: Keras implementation used constraints (MinMaxNorm). 
        # In PyTorch, we usually apply clamping during the forward pass or using a hook/optimizer.
        # For exact parity, we will clamp in forward pass.

    def forward(self, x):
        # x shape: [batch, channel, n_mels, time] or [batch, n_mels, time]
        # Our Conv2DPCEN_TVarying passes [batch, n_mels, time] to each layer.
        
        # Ensure dimensions [batch, n_mels, time]
        if x.dim() == 4:
            x = x.squeeze(1)
            
        # Transpose to [time, batch, n_mels] for loop
        x_t = x.permute(2, 0, 1)
        
        # Smooth coefficient clamping for stability (0, 1)
        s = torch.clamp(self.smooth_coef, 0.001, 0.999)
        
        # IIR Filter: smoothed[t] = (1-s)*x[t] + s*smoothed[t-1]
        smoothed = []
        last_s = torch.zeros_like(x_t[0])
        
        # Manual loop for recurrence (PyTorch jit script could optimize this, but keeping it simple for now)
        for t in range(x_t.size(0)):
            current_s = (1 - s) * x_t[t] + s * last_s
            smoothed.append(current_s)
            last_s = current_s
            
        smoothed = torch.stack(smoothed) # [time, batch, n_mels]
        smoothed = smoothed.permute(1, 2, 0) # [batch, n_mels, time]
        
        # Re-ensure x matches dimensions
        # x was [batch, n_mels, time]
        
        # Apply constraints to other parameters in forward to match Keras constraint behavior roughly
        # Or just use them raw. Keras MinMaxNorm creates hard constraints. 
        # We will assume raw for now unless training becomes unstable, but let's clamp floor.
        
        alpha = self.alpha
        delta = self.delta
        root = self.root
        
        # PCEN: (inputs / (eps + smoothed)^alpha + delta)^root - delta^root
        
        # Add epsilon/floor
        inputs = x
        eps = self.floor
        
        # Term 1: (eps + smoothed)^alpha
        # Note: To match Keras pow(smoothed, -alpha) * inputs, which is inputs / smoothed^alpha
        # We do: inputs * (eps + smoothed)^(-alpha)
        
        smooth_term = (eps + smoothed).pow(-alpha)
        inner_term = inputs * smooth_term + delta
        pcen = inner_term.pow(root) - delta.pow(root)
        
        return pcen


class TVaryingPCEN(nn.Module):
    def __init__(self, n_t_constants=8, trainable=True):
        super(TVaryingPCEN, self).__init__()
        self.n_t_constants = n_t_constants
        self.pcen_layers = nn.ModuleList()
        self.trainable = trainable

        # Create multiple PCEN layers with varying initial smoothing coefficients
        for i in range(n_t_constants):
            smooth_coef = 1.0 / (2 ** i)
            if smooth_coef > 0.99: smooth_coef = 0.99

            # Use our custom PyTorchPCENLayer
            pcen = PyTorchPCENLayer(
                alpha=0.98,
                smooth_coef=smooth_coef,
                delta=1.0, # Using defaults, or user can tune in init
                root=0.5,
                floor=1e-6,
                trainable=trainable
            )
            self.pcen_layers.append(pcen)

    def forward(self, x):
        # x: [batch, n_mels, time]
        
        pcen_outputs = []
        for pcen_layer in self.pcen_layers:
            out = pcen_layer(x) # [batch, n_mels, time]
            pcen_outputs.append(out)

        # Stack along channel dimension: [batch, n_t_constants, n_mels, time]
        return torch.stack(pcen_outputs, dim=1)

