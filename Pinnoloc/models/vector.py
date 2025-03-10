import torch.nn as nn
import torch


class StackedVectorModel(nn.Module):
    def __init__(self, 
                 n_layers,
                 d_input,
                 hidden_units,  # List of hidden units
                 d_output, 
                 activation=nn.Tanh,
                 use_batchnorm=False, 
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
                 hidden_units,  # List of hidden units, d_output=1
                 activation=nn.Tanh,
                 use_batchnorm=False, 
                 dropout_rate=0.0,
                 path_loss_exponent=2.0,
                 rss_1m=-50.0):
        super(DistanceModel, self).__init__(n_layers, d_input, hidden_units, 1, activation, use_batchnorm, dropout_rate)

        log10 = torch.log(torch.as_tensor(10.0, dtype=torch.float32))
        k = log10 / (10.0 * path_loss_exponent)
        self.k = nn.Parameter(k.unsqueeze(0), requires_grad=True)


        rss_1m = torch.as_tensor(rss_1m, dtype=torch.float32)
        self.rss_1m = nn.Parameter(rss_1m.unsqueeze(0), requires_grad=True)


class PositionModel(StackedVectorModel):
    def __init__(self, 
                 n_layers,
                 d_input,
                 hidden_units,  # List of hidden units, d_output=2
                 activation=nn.Tanh,
                 use_batchnorm=False, 
                 dropout_rate=0.0,
                 path_loss_exponent=[1.7, 1.7, 1.7, 1.7]
                 ):
        """
        Initialize the PositionModel.
        Inputs:
        - n_layers: Number of hidden layers in the model
        - d_input: Dimension of the input
        - hidden_units: List of hidden units in each layer
        - activation: Activation function
        - use_batchnorm: Use batch normalization
        - dropout_rate: Dropout rate
        - path_loss_exponent: Path loss exponent (list of float of shape (n_anchors,))
        """
        super(PositionModel, self).__init__(n_layers, d_input, hidden_units, 2, activation, use_batchnorm, dropout_rate)

        path_loss_exponent = torch.as_tensor(path_loss_exponent, dtype=torch.float32)
        self.register_buffer('path_loss_exponent', path_loss_exponent)

        log10 = torch.log(torch.as_tensor(10.0, dtype=torch.float32))
        k = log10 / (10.0 * path_loss_exponent)
        k = torch.as_tensor(k, dtype=torch.float32)
        self.k = nn.Parameter(k, requires_grad=True)

    # @property
    # def k(self):
    #     return torch.nn.functional.softplus(self.k_)  # log(1 + exp(k))
    