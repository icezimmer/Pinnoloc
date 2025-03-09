import torch
import math


class PositionLoss(torch.nn.Module):
    def __init__(self,
                 lambda_data, lambda_rss, lambda_azimuth, lambda_bc,
                 n_collocation, n_boundary_collocation,
                 seed=42,
                 mean_input=None, std_input=None, mean_target=None, std_target=None):
        """
        Custom loss function that combines data prediction loss with physics-based loss for RSSI path loss.
        - Input:
            - lambda_data: Weight for data prediction loss.
            - lambda_rss: Weight for physics-based loss.
        """
        if lambda_data < 0.0 or lambda_rss < 0.0 or lambda_bc < 0.0:
            raise ValueError("Lambda weights must be non-negative.")
        super(PositionLoss, self).__init__()
        self.lambda_data = lambda_data
        self.lambda_rss = lambda_rss
        self.lambda_azimuth = lambda_azimuth
        self.lambda_bc = lambda_bc
        self.n_collocation = n_collocation
        self.n_boundary_collocation = n_boundary_collocation
        self.data_loss_fn = torch.nn.MSELoss()

        # Create a dedicated generator with a fixed seed
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

        # Cache for collocation points (and boundary points if needed)
        self._resample_collocation_points = True
        self._cached_collocation = None
        self._cached_boundary = None

        if mean_input is None:
            self.mean_input = 0.0
        else:
            self.mean_input = mean_input
        if std_input is None:
            self.std_input = 1.0
        else:
            self.std_input = std_input

        if mean_target is None:
            self.mean_target = 0.0
        else:
            self.mean_target = mean_target
        if std_target is None:
            self.std_target = 1.0
        else:
            self.std_target = std_target

    def collocation_points(self, min_x, max_x, min_y, max_y,
                           anchor_x, anchor_y,
                           rss_1m, path_loss_exponent,
                           sigma_rss, sigma_aoa,
                           device):
        """
        Generate collocation points for the physics loss representing RSSI with normal distribution.
        - Input:
            - min_x: Minimum x-coordinate of the domain.
            - max_x: Maximum x-coordinate of the domain.
            - min_y: Minimum y-coordinate of the domain.
            - max_y: Maximum y-coordinate of the domain.
            - anchor_x: x-coordinates of the anchors.
            - anchor_y: y-coordinates of the anchors.
            - rss_1m: RSSI at 1 meter distance.
            - path_loss_exponent: Path loss exponent.
            - sigma_rss: Standard deviation of RSSI.
            - sigma_aoa: Standard deviation of azimuth angle.
            - device: Device to run the model on.
        - Output:
            - P_collocation: RSSI values at collocation points.
            - other_collocation: Other features at collocation points
        """

        n_anchors = len(anchor_x)
        x_ratio = math.floor((max_x - min_x) / (max_y - min_y))
        n = int(math.sqrt(self.n_collocation))
        x_collocation = torch.linspace(min_x, max_x, x_ratio * n, device=device)
        y_collocation = torch.linspace(min_y, max_y, n, device=device)

        X, Y = torch.meshgrid(x_collocation, y_collocation, indexing='ij')
        z_collocation = torch.stack([X.flatten(), Y.flatten()], dim=-1)  # (N, 2)
        N = z_collocation.shape[0]

        anchor_z = torch.stack([anchor_x, anchor_y], dim=-1)  # (n_anchors, 2)

        # compute z_collocation - anchor_z
        z_collocation = z_collocation.unsqueeze(1)  # (N, 1, 2)
        anchor_z = anchor_z.unsqueeze(0)  # (1, n_anchors, 2)
        z_collocation = z_collocation - anchor_z  # (N, n_anchors, 2)

        # compute the distance
        r_collocation = torch.norm(z_collocation, p=2, dim=-1)  # (N, n_anchors)
        # compute the RSSI
        P_collocation = rss_1m - 10 * path_loss_exponent * torch.log10(r_collocation)  # (N, n_anchors)
        P_collocation = P_collocation + sigma_rss * torch.randn(N, n_anchors, generator=self.generator, device=device)

        # compute the azimuth angle
        a_collocation = torch.atan2(z_collocation[:,:,1], z_collocation[:,:,0])  # (N, n_anchors)
        a_collocation = a_collocation + torch.deg2rad(torch.as_tensor(sigma_aoa)) * torch.randn(N, n_anchors, generator=self.generator, device=device)

        ux_collocation = torch.cos(a_collocation)  # (N, n_anchors)
        uy_collocation = torch.sin(a_collocation)  # (N, n_anchors)

        # put in a single tensor ux_1, ux_2, ..., ux_n, uy_1, uy_2, ..., uy_n
        other_collocation = torch.cat((ux_collocation, uy_collocation), dim=-1)  # (N, 2 * n_anchors)

        P_collocation = (P_collocation - self.mean_input[:n_anchors]) / self.std_input[:n_anchors]
        other_collocation = (other_collocation - self.mean_input[n_anchors:]) / self.std_input[n_anchors:]

        return P_collocation, other_collocation
    
    def boundary_collocation_points(self, n_features, n_anchors, rss_bc, u_bc, device):
        """
        Generate boundary collocation points for the physics loss representing RSSI with normal distribution.
        - Input:
            - n_features: Number of features in the input.
            - n_anchors: Number of anchors in the system.
            - rss_bc: RSSI values at boundary collocation points.
            - u_bc: Azimuth angle at boundary collocation points.
            - device: Device to run the model on.
        - Output:
            - boundary_collocation: Boundary collocation
        """
        rss_bc = (rss_bc - self.mean_input[:n_anchors]) / self.std_input[:n_anchors]
        u_bc = (u_bc - self.mean_input[n_anchors:]) / self.std_input[n_anchors:]
        P_collocation = rss_bc * torch.ones(self.n_boundary_collocation, n_anchors, device=device, requires_grad=False) + torch.randn(self.n_boundary_collocation, n_anchors, generator=self.generator, device=device, requires_grad=False)
        other_collocation = u_bc * torch.ones(self.n_boundary_collocation, n_features - n_anchors, device=device, requires_grad=False)
        boundary_collocation = torch.cat((P_collocation, other_collocation), dim=-1)
        return boundary_collocation
    
    def resample_collocation_points(self):
        self._resample_collocation_points = True

    def forward(self, model, inputs, targets):
        if self.lambda_rss == 0.0 and self.lambda_azimuth == 0.0 and self.lambda_bc == 0.0:
            output = model(inputs)
            total_loss = self.data_loss_fn(output, targets)
        
        else:
            output = model(inputs)
            data_loss = self.data_loss_fn(output, targets)  # Data prediction loss

            n_anchors = len(model.anchor_x)  # Number of anchors in the system, (h)

            if self._resample_collocation_points or self._cached_collocation is None:
                self._cached_collocation = self.collocation_points(
                    min_x=model.min_x,
                    max_x=model.max_x,
                    min_y=model.min_y,
                    max_y=model.max_y,
                    anchor_x=model.anchor_x,
                    anchor_y=model.anchor_y,
                    rss_1m=model.rss_1m,
                    path_loss_exponent=model.path_loss_exponent,
                    sigma_rss=model.sigma_rss,
                    sigma_aoa=model.sigma_aoa,
                    device=inputs.device)
                
                self._cached_boundary = [
                    self.boundary_collocation_points(
                    n_features=inputs.shape[-1],
                    n_anchors=n_anchors,
                    rss_bc=model.rss_bc[i,:],
                    u_bc=model.u_bc[i,:],
                    device=inputs.device)
                    for i in range(len(model.z_bc))
                ]

                self._resample_collocation_points = False

            # Collocation points for the physics loss representing RSSI with normal distribution
            P_collocation, other_collocation = self._cached_collocation
            P_collocation = P_collocation.detach().clone().requires_grad_(True)
            other_collocation = other_collocation.detach().clone().requires_grad_(False)
            collocation = torch.cat((P_collocation, other_collocation), dim=-1)     
            z_collocation = model(collocation)
            x_collocation = z_collocation[:,0:1]
            y_collocation = z_collocation[:,1:2]

            dx_dP = torch.autograd.grad(
                outputs=x_collocation,
                inputs=P_collocation,
                grad_outputs=torch.ones_like(x_collocation),  # dLoss/dz = 1, dLoss/dP = dLoss/dz * dz/dP
                create_graph=True
            )[0]
            dy_dP = torch.autograd.grad(
                outputs=y_collocation,
                inputs=P_collocation,
                grad_outputs=torch.ones_like(y_collocation),
                create_graph=True
            )[0]

            dx_dP = (self.std_target[0:1] / self.std_input[0:n_anchors]) * dx_dP
            dy_dP = (self.std_target[1:2] / self.std_input[0:n_anchors]) * dy_dP

            x_collocation = self.std_target[0:1] * x_collocation + self.mean_target[0:1]
            y_collocation = self.std_target[1:2] * y_collocation + self.mean_target[1:2]
            x_collocation = x_collocation - model.anchor_x  # (N, n_anchors)
            y_collocation = y_collocation - model.anchor_y  # (N, n_anchors)
            distance_2 = torch.clamp(torch.pow(x_collocation, 2) + torch.pow(y_collocation, 2), min=1e-8)  # (N, n_anchors)

            residual_x = dx_dP + torch.einsum('h,nh->nh', model.k, x_collocation)  # (N, n_anchors)
            residual_y = dy_dP + torch.einsum('h,nh->nh', model.k, y_collocation)  # (N, n_anchors)
            rss_loss = torch.mean(torch.pow(residual_x, 2)) + torch.mean(torch.pow(residual_y, 2))

            # da/dp = d(atan(y/x))/dx * dx/dp + d(atan(y/x))/dy * dy/dp, assuming da/dp = 0
            residual_azimuth = torch.einsum('nh,nh->nh', x_collocation / distance_2, dy_dP) - torch.einsum('nh,nh->nh', y_collocation / distance_2, dx_dP)
            azimuth_loss = torch.mean(torch.pow(residual_azimuth, 2))

            boundary_collocation_list = self._cached_boundary
            boundary_loss = 0.0
            for i in range(len(model.z_bc)):
                boundary_collocation = boundary_collocation_list[i]
                boundary_collocation = boundary_collocation.detach().clone().requires_grad_(False)
                z_boundary_collocation = model(boundary_collocation)
                z_boundary_collocation = self.std_target * z_boundary_collocation + self.mean_target
                residual_boundary = torch.norm(model.z_bc[i,:] - z_boundary_collocation, p=2, dim=1)
                boundary_loss += torch.mean(torch.pow(residual_boundary, 2))
            boundary_loss /= len(model.z_bc)

            # Compute total loss
            total_loss = (self.lambda_data * data_loss) + (self.lambda_rss * rss_loss) + (self.lambda_azimuth * azimuth_loss) + (self.lambda_bc * boundary_loss)
        
        return total_loss
