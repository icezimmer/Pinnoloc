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


# RSS for position (1.2, 1.2). Press Enter to continue...
#    Anchor_ID  Az_Arrival    AoA_Az        RSS
# 0       6501   -0.977384 -0.811677 -67.886005
# 1       6502    2.897247  2.626567 -77.609357
# 2       6503    3.298672  3.504460 -79.856448
# 3       6504   -2.356194 -2.216139 -75.560155

# RSS for position (10.8, 1.2). Press Enter to continue...
#    Anchor_ID  Az_Arrival    AoA_Az        RSS
# 0       6501   -0.157080 -0.452830 -83.048810
# 1       6502    0.244346  0.991215 -79.380896
# 2       6503    4.118977  3.840250 -65.756944
# 3       6504   -0.785398 -0.719309 -71.955013

# RSS for position (10.8, 4.8). Press Enter to continue...
#    Anchor_ID  Az_Arrival    AoA_Az        RSS
# 0       6501    0.157080  0.553053 -84.354286
# 1       6502    0.785398  1.032371 -79.108696
# 2       6503    2.164208  2.313276 -70.819692
# 3       6504   -0.244346 -0.520642 -83.417533

# RSS for position (1.2, 4.8). Press Enter to continue...
#    Anchor_ID  Az_Arrival    AoA_Az        RSS
# 0       6501    0.977384  0.578434 -72.974209
# 1       6502    2.356194  2.290432 -73.773998
# 2       6503    2.984513  2.880647 -72.976773
# 3       6504   -2.897247 -2.439220 -78.672634



# RSS for position (1.2, 1.2). Press Enter to continue...
#    Anchor_ID  Az_Arrival    AoA_Az        RSS
# 0       6501   -0.977384 -0.811677 -67.886005
# 1       6502    2.897247  2.626567 -77.609357
# 2       6503   -2.984513 -2.725219 -79.856448
# 3       6504   -2.356194 -2.216139 -75.560155

# RSS for position (10.8, 1.2). Press Enter to continue...
#    Anchor_ID  Az_Arrival    AoA_Az        RSS
# 0       6501   -0.157080 -0.452830 -83.048810
# 1       6502    0.244346  0.991215 -79.380896
# 2       6503   -2.164208 -2.442936 -65.756944
# 3       6504   -0.785398 -0.719309 -71.955013

# RSS for position (10.8, 4.8). Press Enter to continue...
#    Anchor_ID  Az_Arrival    AoA_Az        RSS
# 0       6501    0.157080  0.553053 -84.354286
# 1       6502    0.785398  1.032371 -79.108696
# 2       6503    2.164208  2.313276 -70.819692
# 3       6504   -0.244346 -0.520642 -83.417533

# RSS for position (1.2, 4.8). Press Enter to continue...
#    Anchor_ID  Az_Arrival    AoA_Az        RSS
# 0       6501    0.977384  0.578434 -72.974209
# 1       6502    2.356194  2.290432 -73.773998
# 2       6503    2.984513  2.757748 -72.976773
# 3       6504   -2.897247 -2.439220 -78.672634


class PositionModel(StackedVectorModel):
    def __init__(self, 
                 n_layers,
                 d_input,
                 hidden_units,  # List of hidden units, d_output=2
                 activation=nn.Tanh,
                 use_batchnorm=False, 
                 dropout_rate=0.0,
                 min_x=1.2,
                 max_x=10.8,
                 min_y=1.2,
                 max_y=4.8,
                 anchor_x=[0.0, 6.0, 12.0, 6.0],
                 anchor_y=[3.0, 0.0, 3.0, 6.0],
                 rss_1m=[-55.0, -55.0, -55.0, -55.0],
                 path_loss_exponent=[1.7, 1.7, 1.7, 1.7],
                 sigma_rss=5.0,
                 sigma_aoa=10.0,
                 z_bc=[[1.2, 1.2],
                      [10.8, 1.2],
                      [10.8, 4.8],
                      [1.2, 4.8]],
                 rss_bc=[[-68.0, -78.0, -80.0, -76.0],
                        [-83.0, -79.0, -66.0, -72.0],
                        [-84.0, -79.0, -71.0, -83.0],
                        [-73.0, -74.0, -73.0, -79.0]],
                 aoa_bc=[[-56, -90+76, -180+9, 90-45],
                      [-9, -90-76, -180+56, 90+45],
                      [9, -90-45, -180-56, 90+76],
                      [56, -90+45, -180-9, 90-76]]
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
        - min_x: Minimum x-coordinate (float)
        - max_x: Maximum x-coordinate (float)
        - min_y: Minimum y-coordinate (float)
        - max_y: Maximum y-coordinate (float)
        - anchor_x: x-coordinate of the anchors (list of float of shape (n_anchors,))
        - anchor_y: y-coordinate of the anchors (list of float of shape (n_anchors,))
        - rss_1m: RSS at 1m from the anchors (list of float of shape (n_anchors,))
        - path_loss_exponent: Path loss exponent (list of float of shape (n_anchors,))
        - sigma_rss: Standard deviation of RSS (float)
        - sigma_aoa: Standard deviation of AoA (float)
        - z_bc: Ground truth positions (list of list of float of shape (n_points, 2))
        - rss_bc: RSS measurements (list of list of float of shape (n_points, n_anchors))
        - aoa_bc: AoA measurements (list of list of float of shape (n_points, n_anchors))
        """
        super(PositionModel, self).__init__(n_layers, d_input, hidden_units, 2, activation, use_batchnorm, dropout_rate)

        self.min_x, self.max_x = min_x, max_x
        self.min_y, self.max_y = min_y, max_y

        self.sigma_rss = sigma_rss
        self.sigma_aoa = sigma_aoa
        
        anchor_x = torch.as_tensor(anchor_x, dtype=torch.float32)
        self.register_buffer('anchor_x', anchor_x)

        anchor_y = torch.as_tensor(anchor_y, dtype=torch.float32)
        self.register_buffer('anchor_y', anchor_y)

        rss_1m = torch.as_tensor(rss_1m, dtype=torch.float32)
        self.register_buffer('rss_1m', rss_1m)

        path_loss_exponent = torch.as_tensor(path_loss_exponent, dtype=torch.float32)
        self.register_buffer('path_loss_exponent', path_loss_exponent)

        log10 = torch.log(torch.as_tensor(10.0, dtype=torch.float32))
        k = log10 / (10.0 * path_loss_exponent)
        k = torch.as_tensor(k, dtype=torch.float32)
        self.k = nn.Parameter(k, requires_grad=True)

        rss_bc = torch.as_tensor(rss_bc, dtype=torch.float32)
        self.register_buffer('rss_bc', rss_bc)

        aoa_bc = torch.as_tensor(aoa_bc, dtype=torch.float32)
        aoa_bc = torch.deg2rad(aoa_bc)
        ux_bc = torch.cos(aoa_bc)
        uy_bc = torch.sin(aoa_bc)
        u_bc = torch.cat((ux_bc, uy_bc), dim=-1)
        self.register_buffer('u_bc', u_bc)

        z_bc = torch.as_tensor(z_bc, dtype=torch.float32)
        self.z_bc = nn.Parameter(z_bc, requires_grad=True)

    # @property
    # def k(self):
    #     return torch.nn.functional.softplus(self.k_)  # log(1 + exp(k))
    