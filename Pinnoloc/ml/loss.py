import torch


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

    def collocation_points(self, anchor_x, anchor_y, path_loss_exponent, device, requires_grad=True):
        """
        Generate collocation points for the physics loss representing RSSI with normal distribution.
        - Input:
            - n_collocation: Number of collocation points.
            - n_features: Number of features in the input data.
            - n_anchors: Number of anchors in the system.
            - device: Device to run the model on.
            - requires_grad: Whether to compute gradients.
        - Output:
            - P_collocation: Collocation points for the RSSI.
            - other_collocation: Collocation points for the azimuth.
        """
        # P_collocation = torch.randn(self.n_collocation, n_anchors, generator=self.generator, device=device, requires_grad=requires_grad)
        # other_collocation = torch.randn(self.n_collocation, n_features - n_anchors, generator=self.generator, device=device, requires_grad=False)
        # P_collocation = -3.0 + 6.0 * torch.rand(self.n_collocation, n_anchors, generator=self.generator, device=device, requires_grad=requires_grad)
        # other_collocation = -3.0 + 6.0 * torch.rand(self.n_collocation,n_features - n_anchors, generator=self.generator, device=device, requires_grad=False)

        n_anchors = len(anchor_x)
        x_collocation = torch.linspace(1.2, 10.8, 100 * 2 , device=device) # (n_collocation)
        y_collocation = torch.linspace(1.2, 4.8, 100, device=device) # (n_collocation)

        X, Y = torch.meshgrid(x_collocation, y_collocation, indexing='ij')
        z_collocation = torch.stack([X.flatten(), Y.flatten()], dim=-1)  # (N, 2)

        anchor_z = torch.stack([anchor_x, anchor_y], dim=-1)  # (n_anchors, 2)

        # compute z_collocation - anchor_z
        z_collocation = z_collocation.unsqueeze(1)  # (N, 1, 2)

        # import matplotlib.pyplot as plt
        # plt.scatter(z_collocation[:,0,0].flatten().cpu().numpy(), z_collocation[:,0,1].flatten().cpu().numpy())
        # plt.show()

        # anchor_z = anchor_z.unsqueeze(0)  # (1, n_anchors, 2)
        z_collocation = z_collocation - anchor_z.unsqueeze(0)  # (N, n_anchors, 2)

        # plt.scatter(z_collocation[:,0,0].flatten().cpu().numpy(), z_collocation[:,0,1].flatten().cpu().numpy(), label='6501')
        # plt.scatter(z_collocation[:,1,0].flatten().cpu().numpy(), z_collocation[:,1,1].flatten().cpu().numpy(), label='6502')
        # plt.scatter(z_collocation[:,2,0].flatten().cpu().numpy(), z_collocation[:,2,1].flatten().cpu().numpy(), label='6503')
        # plt.scatter(z_collocation[:,3,0].flatten().cpu().numpy(), z_collocation[:,3,1].flatten().cpu().numpy(), label='6504')
        # plt.legend()
        # plt.show()

        # compute the azimuth angle
        a_collocation = torch.atan2(z_collocation[:,:,1], z_collocation[:,:,0])  # (N, n_anchors)

        # compute the distance
        r_collocation = torch.norm(z_collocation, p=2, dim=-1)  # (N, n_anchors)

        # compute the RSSI
        # p = rss_0 - 10 * k * torch.log10(r_collocation)
        # k = torch.as_tensor([1.52, 2.1, 1.53, 1.63], device=device)
        # rss_0 = torch.as_tensor([-58, -59, -58, -58], device=device)
        k = path_loss_exponent
        rss_0 = torch.as_tensor([-58, -58, -58, -58], device=device)
        P_collocation = rss_0.unsqueeze(0) - 10 * k.unsqueeze(0) * torch.log10(r_collocation)  # (N, n_anchors)

        a_collocation = (a_collocation - self.mean_input[n_anchors:]) / self.std_input[n_anchors:]
        P_collocation = (P_collocation - self.mean_input[:n_anchors]) / self.std_input[:n_anchors]

        # P_collocation requires gradient
        P_collocation = P_collocation.detach().clone().requires_grad_(True)

        return P_collocation, a_collocation
        # return P_collocation, other_collocation
    
    # def boundary_collocation_points(self, n_features, n_anchors, device):
    #     """
    #     Generate boundary collocation points for the physics loss representing RSSI with normal distribution.
    #     - Input:
    #         - n_boundary_collocation: Number of boundary collocation points.
    #         - n_features: Number of features in the input data.
    #         - n_anchors: Number of anchors in the system.
    #         - device: Device to run the model on.
    #     - Output:
    #         - boundary_collocation: Boundary collocation points for a particular position
    #     """
    #     a_0 = torch.as_tensor([0.0, torch.pi/2, torch.pi, -torch.pi/2]) * torch.ones(self.n_boundary_collocation, n_anchors, device=device, requires_grad=False)
    #     other_collocation = torch.randn(self.n_boundary_collocation, n_features - n_anchors, generator=self.generator, device=device, requires_grad=False)
    #     boundary_collocation = torch.cat((other_collocation, a_0), dim=-1)
    #     return boundary_collocation
    
    def boundary_collocation_points(self, n_features, n_anchors, rss_0, a_0, device):
        """
        Generate boundary collocation points for the physics loss representing RSSI with normal distribution.
        - Input:
            - n_boundary_collocation: Number of boundary collocation points.
            - n_features: Number of features in the input data.
            - n_anchors: Number of anchors in the system.
            - device: Device to run the model on.
        - Output:
            - boundary_collocation: Boundary collocation points for a particular position
        """
        rss_0 = (rss_0 - self.mean_input[:n_anchors]) / self.std_input[:n_anchors]
        a_0 = (a_0 - self.mean_input[n_anchors:]) / self.std_input[n_anchors:]
        P_collocation = rss_0 * torch.ones(self.n_boundary_collocation, n_anchors, device=device, requires_grad=False) + torch.randn(self.n_boundary_collocation, n_anchors, generator=self.generator, device=device, requires_grad=False)
        other_collocation = a_0 * torch.ones(self.n_boundary_collocation, n_features - n_anchors, device=device, requires_grad=False)
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
                    anchor_x=model.anchor_x,
                    anchor_y=model.anchor_y,
                    path_loss_exponent=model.path_loss_exponent,
                    device=inputs.device,
                    requires_grad=True)
                
                self._cached_boundary = [
                    self.boundary_collocation_points(
                    n_features=inputs.shape[-1],
                    n_anchors=n_anchors,
                    rss_0=model.rss_0[i,:],
                    a_0=model.a_0[i,:],
                    device=inputs.device)
                    for i in range(len(model.z_0))
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
            # a_collocation = torch.atan((y_collocation - ((model.anchor_y - self.mean_target[1:2]) / self.std_target[1:2])) / (x_collocation - ((model.anchor_x - self.mean_target[0:1]) / self.std_target[0:1])))

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

            residual_x = dx_dP + torch.einsum('h,nh->nh', model.k, x_collocation)
            residual_y = dy_dP + torch.einsum('h,nh->nh', model.k, y_collocation)
            rss_loss = torch.mean(torch.pow(residual_x, 2)) + torch.mean(torch.pow(residual_y, 2))

            a_collocation = torch.atan2(y_collocation, x_collocation)
            a_collocation = (a_collocation - self.mean_input[n_anchors:]) / self.std_input[n_anchors:]
            azimuth_loss = 0.0
            for anchor_id in range(n_anchors):
                da_dP = torch.autograd.grad(
                    outputs=a_collocation[:,anchor_id:anchor_id+1],
                    inputs=P_collocation,
                    grad_outputs=torch.ones_like(a_collocation[:,anchor_id:anchor_id+1]),
                    create_graph=True
                )[0]
                da_dP = (self.std_input[n_anchors+anchor_id:5+anchor_id] / self.std_input[0:n_anchors]) * da_dP
                residual_azimuth = da_dP - torch.einsum('nh,nh->nh', x_collocation / distance_2, dy_dP) + torch.einsum('nh,nh->nh', y_collocation / distance_2, dx_dP)
                azimuth_loss += torch.mean(torch.pow(residual_azimuth, 2))
            azimuth_loss /= n_anchors

            # boundary_collocation = self._cached_boundary
            # boundary_collocation = boundary_collocation.detach().clone().requires_grad_(False)
            # z_boundary_collocation = model(boundary_collocation)
            # z_boundary_collocation = self.std_target * z_boundary_collocation + self.mean_target
            # residual_boundary = model.z_0 - z_boundary_collocation
            # boundary_loss = torch.mean(torch.pow(residual_boundary, 2))

            boundary_collocation_list = self._cached_boundary
            boundary_loss = 0.0
            for i in range(len(model.z_0)):
                boundary_collocation = boundary_collocation_list[i]
                boundary_collocation = boundary_collocation.detach().clone().requires_grad_(False)
                z_boundary_collocation = model(boundary_collocation)
                z_boundary_collocation = self.std_target * z_boundary_collocation + self.mean_target
                residual_boundary = torch.norm(model.z_0[i,:] - z_boundary_collocation, p=2, dim=1)
                boundary_loss += torch.mean(torch.pow(residual_boundary, 2))
            boundary_loss /= len(model.z_0)

            # Compute total loss
            total_loss = (self.lambda_data * data_loss) + (self.lambda_rss * rss_loss) + (self.lambda_azimuth * azimuth_loss) + (self.lambda_bc * boundary_loss)
        
        return total_loss
