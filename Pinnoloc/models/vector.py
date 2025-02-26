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
                 path_loss=2.0,
                 rss_1m=-50.0):
        super(DistanceModel, self).__init__(n_layers, d_input, hidden_units, 1, activation, use_batchnorm, dropout_rate)

        log10 = torch.log(torch.as_tensor(10.0, dtype=torch.float32))
        k = log10 / (10.0 * path_loss)
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
                 anchor_x=0.0,
                 anchor_y=0.0,
                 path_loss=2.0,
                 rss_1m=-50.0):
        super(PositionModel, self).__init__(n_layers, d_input, hidden_units, 2, activation, use_batchnorm, dropout_rate)

        # Define buffer of model
        anchor_x = torch.as_tensor(anchor_x, dtype=torch.float32)
        self.register_buffer('anchor_x', anchor_x.unsqueeze(0))

        anchor_y = torch.as_tensor(anchor_y, dtype=torch.float32)
        self.register_buffer('anchor_y', anchor_y.unsqueeze(0))

        log10 = torch.log(torch.as_tensor(10.0, dtype=torch.float32))
        k = log10 / (10.0 * path_loss)
        self.k = nn.Parameter(k.unsqueeze(0), requires_grad=True)


        rss_1m = torch.as_tensor(rss_1m, dtype=torch.float32)
        self.rss_1m = nn.Parameter(rss_1m.unsqueeze(0), requires_grad=False)