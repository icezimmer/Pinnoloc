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
                 anchor_x=[0.0, 6.0, 12.0, 6.0],
                 anchor_y=[3.0, 0.0, 3.0, 6.0],
                 path_loss_exponent=[2.0, 2.0, 2.0, 2.0],
                 rss_1m=[-50.0, -50.0, -50.0, -50.0]):
        super(PositionModel, self).__init__(n_layers, d_input, hidden_units, 2, activation, use_batchnorm, dropout_rate)

        # Define buffer of model
        anchor_x = torch.as_tensor(anchor_x, dtype=torch.float32)
        self.register_buffer('anchor_x', anchor_x)

        anchor_y = torch.as_tensor(anchor_y, dtype=torch.float32)
        self.register_buffer('anchor_y', anchor_y)

        path_loss_exponent = torch.as_tensor(path_loss_exponent, dtype=torch.float32)
        log10 = torch.log(torch.as_tensor(10.0, dtype=torch.float32))
        k = log10 / (10.0 * path_loss_exponent)
        k = torch.as_tensor(k, dtype=torch.float32)
        print(k)
        self.k = nn.Parameter(k, requires_grad=True)

        rss_1m = torch.as_tensor(rss_1m, dtype=torch.float32)
        # self.rss_1m = nn.Parameter(rss_1m, requires_grad=True)
        self.register_buffer('rss_1m', rss_1m)

        d_0 = torch.as_tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
        self.d_0 = nn.Parameter(d_0, requires_grad=True)

    # @property
    # def k(self):
    #     return torch.nn.functional.softplus(self.k_)  # log(1 + exp(k))
    
    # @property
    # def rss_1m(self):
    #     return -torch.nn.functional.softplus(-self.rss_1m_)
    
    # @property
    # def d_0(self):
    #     return torch.nn.functional.softplus(self.d_0_)
    