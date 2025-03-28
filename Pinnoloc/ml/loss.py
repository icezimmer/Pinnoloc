import torch
import math


class PositionLoss(torch.nn.Module):
    def __init__(self,
                 lambda_data, lambda_physics, lambda_bc,
                 n_collocation, n_boundary_collocation,
                 min_x=0.0,
                 max_x=12.0,
                 min_y=0.0,
                 max_y=6.0,
                 anchor_x=[0.0, 6.0, 12.0, 6.0],
                 anchor_y=[3.0, 0.0, 3.0, 6.0],
                 rss_1m=[-58.0, -58.0, -58.0, -58.0],
                 path_loss_exponent=[1.69, 1.69, 1.69, 1.69],
                 sigma_rss=2.0,
                 sigma_aoa=5.0,
                 seed=42,
                 mean_input=None, std_input=None, mean_target=None, std_target=None):
        """
        Custom loss function that combines data prediction loss with physics-based loss for RSSI path loss.
        - Input:
            - lambda_data: Weight for data prediction loss.
            - lambda_physics: Weight for physics-based loss.
            - lambda_bc: Weight for boundary condition loss.
            - n_collocation: Number of collocation points for the physics-based loss.
            - n_boundary_collocation: Number of boundary collocation points.
            - min_x: Minimum x-coordinate (float)
            - max_x: Maximum x-coordinate (float)
            - min_y: Minimum y-coordinate (float)
            - max_y: Maximum y-coordinate (float)
            - anchor_x: x-coordinate of the anchors (list of float of shape (n_anchors,))
            - anchor_y: y-coordinate of the anchors (list of float of shape (n_anchors,))
            - rss_1m: RSS at 1m from the anchors (list of float of shape (n_anchors,))
            - path_loss_exponent: Path loss exponent (list of float of shape (n_anchors,))
            - sigma_rss: Standard deviation of the RSSI noise (float dB).
            - sigma_aoa: Standard deviation of the AoA noise (float degrees).
            - seed: Seed for the random generator (int).
            - mean_input: Mean of the input features (tensor of shape (3 * n_anchors,)).
            - std_input: Standard deviation of the input features (tensor of shape (3 * n_anchors,)).
            - mean_target: Mean of the target features (tensor of shape (2,)).
            - std_target: Standard deviation of the target features (tensor of shape (2,)).
        """
        if lambda_data < 0.0 or lambda_physics < 0.0 or lambda_bc < 0.0:
            raise ValueError("Lambda weights must be non-negative.")
            
        super(PositionLoss, self).__init__()
        self.lambda_data = lambda_data
        self.lambda_physics = lambda_physics
        self.lambda_bc = lambda_bc
        self.n_collocation = n_collocation
        self.n_boundary_collocation = n_boundary_collocation
        self.data_loss_fn = torch.nn.MSELoss()

        anchor_x = torch.as_tensor(anchor_x, dtype=torch.float32)
        anchor_y = torch.as_tensor(anchor_y, dtype=torch.float32)
        path_loss_exponent = torch.as_tensor(path_loss_exponent, dtype=torch.float32)
        rss_1m = torch.as_tensor(rss_1m, dtype=torch.float32)
        self.register_buffer('anchor_x', anchor_x)
        self.register_buffer('anchor_y', anchor_y)
        self.register_buffer('rss_1m', rss_1m)
        self.register_buffer('path_loss_exponent', path_loss_exponent)

        self.sigma_rss = sigma_rss
        self.sigma_aoa = sigma_aoa

        P_collocation_grid, a_collocation_grid = self._collocation_grid(min_x, max_x, min_y, max_y)
        self.register_buffer('P_collocation_grid', P_collocation_grid)
        self.register_buffer('a_collocation_grid', a_collocation_grid)

        P_collocation_walls, a_collocation_walls, target_collocation_walls = self._collocation_walls(min_x, max_x, min_y, max_y)
        self.register_buffer('P_collocation_walls', P_collocation_walls)
        self.register_buffer('a_collocation_walls', a_collocation_walls)
        self.register_buffer('target_collocation_walls', target_collocation_walls)

        # Create a dedicated generator with a fixed seed
        self.seed = seed
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

        # Cache for collocation points (and boundary points if needed)
        self._resample_collocation_points = True
        self._cached_collocation = None
        self._cached_boundary = None

        if mean_input is None:
            mean_input = torch.zeros(3 * len(anchor_x), dtype=torch.float32)
        if std_input is None:
            std_input = torch.ones(3 * len(anchor_x), dtype=torch.float32)
        self.register_buffer('mean_input', mean_input)
        self.register_buffer('std_input', std_input)

        if mean_target is None:
            mean_target = torch.zeros(2, dtype=torch.float32)
        if std_target is None:
            std_target = torch.ones(2, dtype=torch.float32)
        self.register_buffer('mean_target', mean_target)
        self.register_buffer('std_target', std_target)

    def to(self, *args, **kwargs):
        """Ensure generator moves with the model."""
        device = torch._C._nn._parse_to(*args, **kwargs)[0]  # Extract target device
        
        # Recreate the generator on the new device with the same seed
        self.generator = torch.Generator(device).manual_seed(self.seed)
        
        return super().to(*args, **kwargs)

    def _collocation_grid(self, min_x, max_x, min_y, max_y):
        """
        Generate collocation points for the physics loss representing RSSI and azimuth angles at grid points.
        - Input:
            - min_x: Minimum x-coordinate of the domain.
            - max_x: Maximum x-coordinate of the domain.
            - min_y: Minimum y-coordinate of the domain.
            - max_y: Maximum y-coordinate of the domain.
        - Output:
            - P_collocation: RSSI values at collocation points.
            - other_collocation: Other features at collocation points
        """

        # Estimate the number of points in x and y directions
        n_x = round((self.n_collocation * ((max_x - min_x) / (max_y - min_y))) ** 0.5)
        n_y = round(self.n_collocation / n_x)

        # Create evenly spaced points along x and y
        x_collocation = torch.linspace(min_x, max_x, n_x + 2, dtype=torch.float32)
        y_collocation = torch.linspace(min_y, max_y, n_y + 2, dtype=torch.float32)
        # leave the endpoints out
        x_collocation = x_collocation[1:-1]
        y_collocation = y_collocation[1:-1]

        X, Y = torch.meshgrid(x_collocation, y_collocation, indexing='ij')
        z_collocation = torch.stack([X.flatten(), Y.flatten()], dim=-1)  # (N, 2)

        anchor_z = torch.stack([self.anchor_x, self.anchor_y], dim=-1)  # (n_anchors, 2)
        # compute differences z_collocation - anchor_z
        z_collocation = z_collocation.unsqueeze(1)  # (N, 1, 2)
        anchor_z = anchor_z.unsqueeze(0)  # (1, n_anchors, 2)
        z_collocation = z_collocation - anchor_z  # (N, n_anchors, 2)

        # compute the distances
        r_collocation = torch.norm(z_collocation, p=2, dim=-1)  # (N, n_anchors)
        # compute the RSSIs
        P_collocation = self.rss_1m - 10 * self.path_loss_exponent * torch.log10(r_collocation + 1e-8)  # (N, n_anchors)

        # compute the azimuth angles
        a_collocation = torch.atan2(z_collocation[:,:,1], z_collocation[:,:,0])  # (N, n_anchors)

        return P_collocation, a_collocation
    
    def _collocation_walls(self, min_x, max_x, min_y, max_y):
        n_x = round((self.n_boundary_collocation * ((max_x - min_x) / (max_y - min_y))) ** 0.5)
        n_y = round(self.n_boundary_collocation / n_x)

        # create points on the walls
        x_collocation = torch.linspace(min_x, max_x, n_x, dtype=torch.float32)
        y_collocation = torch.linspace(min_y, max_y, n_y, dtype=torch.float32)

        X1, Y1 = torch.meshgrid(torch.tensor([min_x]), y_collocation, indexing='ij')
        X2, Y2 = torch.meshgrid(x_collocation, torch.tensor([min_y]), indexing='ij')
        X3, Y3 = torch.meshgrid(torch.tensor([max_x]), y_collocation, indexing='ij')
        X4, Y4 = torch.meshgrid(x_collocation, torch.tensor([max_y]), indexing='ij')

        X, Y = torch.cat((X1.flatten(), X2.flatten(), X3.flatten(), X4.flatten())), torch.cat((Y1.flatten(), Y2.flatten(), Y3.flatten(), Y4.flatten()))
        z_collocation = torch.stack([X, Y], dim=-1)  # (N, 2)
        target_collocation = z_collocation

        anchor_z = torch.stack([self.anchor_x, self.anchor_y], dim=-1)  # (n_anchors, 2)
        # compute differences z_collocation - anchor_z
        z_collocation = z_collocation.unsqueeze(1)  # (N, 1, 2)
        anchor_z = anchor_z.unsqueeze(0)  # (1, n_anchors, 2)
        z_collocation = z_collocation - anchor_z  # (N, n_anchors, 2)

        # compute the distances
        r_collocation = torch.norm(z_collocation, p=2, dim=-1)  # (N, n_anchors)
        # compute the RSSIs
        P_collocation = self.rss_1m - 10 * self.path_loss_exponent * torch.log10(r_collocation + 1e-8)  # (N, n_anchors)

        # compute the azimuth angles
        a_collocation = torch.atan2(z_collocation[:,:,1], z_collocation[:,:,0])  # (N, n_anchors)

        return P_collocation, a_collocation, target_collocation


    def _collocation_points(self, device):
        """
        Generate collocation points for the physics loss representing RSSI and azimuth versors with gaussian noise.
        - Input:
            - device: Device to run the model on.
        - Output:
            - P_collocation: RSSI values at collocation points.
            - other_collocation: Other features at collocation points
        """
        N, n_anchors = self.P_collocation_grid.shape

        P_collocation = self.P_collocation_grid + self.sigma_rss * torch.randn(N, n_anchors, generator=self.generator, device=device, dtype=torch.float32)
        a_collocation = self.a_collocation_grid + torch.deg2rad(torch.as_tensor(self.sigma_aoa, dtype=torch.float32)) * torch.randn(N, n_anchors, generator=self.generator, device=device, dtype=torch.float32)

        ux_collocation = torch.cos(a_collocation)  # (N, n_anchors)
        uy_collocation = torch.sin(a_collocation)  # (N, n_anchors)

        # put in a single tensor ux_1, ux_2, ..., ux_n, uy_1, uy_2, ..., uy_n
        other_collocation = torch.cat((ux_collocation, uy_collocation), dim=-1)  # (N, 2 * n_anchors)

        P_collocation = (P_collocation - self.mean_input[:n_anchors]) / self.std_input[:n_anchors]
        other_collocation = (other_collocation - self.mean_input[n_anchors:]) / self.std_input[n_anchors:]

        return P_collocation, other_collocation
    
    def _collocation_boundary(self, device):
        N, n_anchors = self.P_collocation_walls.shape

        P_collocation = self.P_collocation_walls + self.sigma_rss * torch.randn(N, n_anchors, generator=self.generator, device=device, dtype=torch.float32)
        a_collocation = self.a_collocation_walls  # no noise for the azimuth angles


        ux_collocation = torch.cos(a_collocation)  # (N, n_anchors)
        uy_collocation = torch.sin(a_collocation)  # (N, n_anchors)

        # put in a single tensor P_1, P_2, ..., P_n, ux_1, ux_2, ..., ux_n, uy_1, uy_2, ..., uy_n
        input_collocation = torch.cat((P_collocation, ux_collocation, uy_collocation), dim=-1)  # (N, 3 * n_anchors)

        input_collocation = (input_collocation - self.mean_input) / self.std_input
        target_collocation = self.target_collocation_walls
        target_collocation = (target_collocation - self.mean_target) / self.std_target

        return input_collocation, target_collocation
    
    def resample_collocation_points(self):
        self._resample_collocation_points = True

    def forward(self, model, inputs, targets):
        if self.lambda_data > 0.0:
            output = model(inputs)
            data_loss = self.data_loss_fn(output, targets)
            total_loss = self.lambda_data * data_loss
        else:
            total_loss = 0.0
        
        if self.lambda_physics > 0.0 or self.lambda_bc > 0.0:

            n_anchors = len(self.anchor_x)  # Number of anchors in the system, (h)

            if self._resample_collocation_points or self._cached_collocation is None:
                self._cached_collocation = self._collocation_points(device=inputs.device)
                self._cached_boundary = self._collocation_boundary(device=inputs.device)
                self._resample_collocation_points = False

            # Collocation points for the physics loss representing RSSI with normal distribution
            P_collocation, other_collocation = self._cached_collocation
            P_collocation = P_collocation.detach().clone().requires_grad_(True)
            other_collocation = other_collocation.detach().clone().requires_grad_(False)
            collocation = torch.cat((P_collocation, other_collocation), dim=-1)     
            z_collocation = model(collocation)  # (N, 2)
            x_collocation = z_collocation[:,0:1]  # (N, 1)
            y_collocation = z_collocation[:,1:2]  # (N, 1)

            dx_dP = torch.autograd.grad(
                outputs=x_collocation,
                inputs=P_collocation,
                grad_outputs=torch.ones_like(x_collocation, dtype=torch.float32),  # dLoss/dz = 1, dLoss/dP = dLoss/dz * dz/dP
                create_graph=True
            )[0]  # (N, n_anchors)
            dy_dP = torch.autograd.grad(
                outputs=y_collocation,
                inputs=P_collocation,
                grad_outputs=torch.ones_like(y_collocation, dtype=torch.float32),
                create_graph=True
            )[0]  # (N, n_anchors)

            dx_dP = (self.std_target[0:1] / self.std_input[0:n_anchors]) * dx_dP  # (N, n_anchors)
            dy_dP = (self.std_target[1:2] / self.std_input[0:n_anchors]) * dy_dP  # (N, n_anchors)

            x_collocation = self.std_target[0:1] * x_collocation + self.mean_target[0:1]  # (N, 1)
            y_collocation = self.std_target[1:2] * y_collocation + self.mean_target[1:2]  # (N, 1)
            x_collocation = x_collocation - self.anchor_x  # (N, n_anchors)
            y_collocation = y_collocation - self.anchor_y  # (N, n_anchors)
            # distance_2 = torch.clamp(torch.pow(x_collocation, 2) + torch.pow(y_collocation, 2), min=1e-8)  # (N, n_anchors)

            residual_x = dx_dP + model.k * x_collocation  # (N, n_anchors)
            residual_y = dy_dP + model.k * y_collocation  # (N, n_anchors)
            rss_x_loss = torch.mean(torch.pow(residual_x, 2))
            rss_y_loss = torch.mean(torch.pow(residual_y, 2))
            physics_loss = (rss_x_loss + rss_y_loss) / 2

            input_boundary, target_boundary = self._cached_boundary
            predicted_boundary = model(input_boundary)
            boundary_loss = self.data_loss_fn(predicted_boundary, target_boundary)

            # Compute total loss
            total_loss += self.lambda_physics * physics_loss + self.lambda_bc * boundary_loss
        
        return total_loss
