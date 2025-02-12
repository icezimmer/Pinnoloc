import torch.nn as nn
import torch
    

class StackedImageModel(nn.Module):
    def __init__(self,
                 n_layers,
                 input_height,
                 input_width,
                 input_channels,
                 filters,  # List of filter sizes
                 d_output=10,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 use_batchnorm=True, 
                 dropout_rate=0.0,
                 pool_every=1):  # Pool every N layers
        super(StackedImageModel, self).__init__()
        
        layers = []
        in_channels = input_channels
        
        for i in range(n_layers):
            out_channels = filters[i] if i < len(filters) else filters[-1]  # Expand last filter size if needed
            
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding))
            layers.append(nn.ReLU(inplace=True))
            
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(out_channels))
            
            # Apply pooling every `pool_every` layers
            if (i + 1) % pool_every == 0:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))  
            
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Compute feature map size dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, input_height, input_width)
            conv_out = self.conv_layers(dummy_input)
            feature_map_size = conv_out.numel() // conv_out.shape[0]  # C * H * W
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(feature_map_size, filters[-1]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(filters[-1], d_output)
        )
        
    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, start_dim=1)  # Flatten before feeding into FC
        x = self.fc(x)
        return x
