import torch.nn as nn
import torch


class StackedVectorModel(nn.Module):
    def __init__(self, 
                 n_layers,
                 d_input,
                 hidden_units,  # List of hidden units
                 d_output, 
                 activation=nn.ReLU,
                 use_batchnorm=True, 
                 dropout_rate=0.0):
        super(StackedVectorModel, self).__init__()
        
        layers = []
        in_features = d_input
        
        for i in range(n_layers):
            out_features = hidden_units[i] if i < len(hidden_units) else hidden_units[-1]
            layers.append(nn.Linear(in_features, out_features))
            
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(out_features))
            
            layers.append(activation())
            layers.append(nn.Dropout(dropout_rate))
            in_features = out_features
        
        layers.append(nn.Linear(in_features, d_output))
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)  # Flatten for FC layers
        x = self.mlp(x)
        return x


# Define a subclass of the StackedVectorModel that uses learnable path loss
class DistanceModel(StackedVectorModel):
    def __init__(self, 
                 n_layers,
                 d_input,
                 hidden_units,  # List of hidden units
                 d_output, 
                 activation=nn.ReLU,
                 use_batchnorm=True, 
                 dropout_rate=0.0):
        super(DistanceModel, self).__init__(n_layers, d_input, hidden_units, d_output, activation, use_batchnorm, dropout_rate)

        self.P_collocation = torch.randn(1000, 1, requires_grad=True)
        self.path_loss = nn.Parameter(torch.ones(d_input), requires_grad=True)
