import torch


class PositionLoss(torch.nn.Module):
    def __init__(self, lambda_data, lambda_rss, lambda_azimuth, lambda_bc, n_collocation=1000,
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
        self.data_loss_fn = torch.nn.MSELoss()

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

    @staticmethod
    def collocation_points(n_collocation, n_features, n_anchors, device, requires_grad=True):
        # Collocation points for the physics loss representing RSSI with normal distribution
        P_collocation = torch.randn(n_collocation, n_anchors, device=device, requires_grad=requires_grad)
        other_collocation = torch.randn(n_collocation, n_features - n_anchors, device=device, requires_grad=False)
        # P_collocation = -3.0 + 6.0 * torch.rand(n_collocation, 1, device=device, requires_grad=requires_grad)
        # other_collocation = -3.0 + 6.0 * torch.rand(n_collocation, n_features - 1, device=device, requires_grad=False)
        collocation = torch.cat((P_collocation, other_collocation), dim=-1)
        return P_collocation, collocation
    
    @staticmethod
    def boundary_collocation_points(n_collocation, n_features, n_anchors, rss_1m, mean_input, std_input, device):
        # Collocation points for the physics loss representing RSSI with normal distribution
        P0 = (rss_1m - mean_input[0:n_anchors]) / std_input[0:n_anchors]  # standardized RSSI
        P0 = torch.einsum('h,nh->nh', P0, torch.ones(n_collocation, n_anchors, device=device, requires_grad=False))
        other_collocation = torch.randn(n_collocation, n_features - n_anchors, device=device, requires_grad=False)
        # other_collocation = -3.0 + 6.0 * torch.rand(n_collocation, n_features - 1, device=device, requires_grad=False)
        boundary_collocation = torch.cat((P0, other_collocation), dim=-1)
        return boundary_collocation

    def forward(self, model, inputs, targets):
        if self.lambda_rss == 0.0 and self.lambda_azimuth == 0.0 and self.lambda_bc == 0.0:
            output = model(inputs)
            total_loss = self.data_loss_fn(output, targets)
        
        else:
            output = model(inputs)
            data_loss = self.data_loss_fn(output, targets)  # Data prediction loss

            # Collocation points for the physics loss representing RSSI with normal distribution
            n_anchors = len(model.anchor_x)  # Number of anchors in the system, (h)
            P_collocation, collocation = self.collocation_points(n_collocation=self.n_collocation,
                                                                 n_features=inputs.shape[-1], n_anchors=n_anchors,
                                                                 device=inputs.device, requires_grad=True)
            
            z_collocation = model(collocation)
            x_collocation = z_collocation[:,0:1]
            y_collocation = z_collocation[:,1:2]
            a_collocation = torch.atan((y_collocation - ((model.anchor_y - self.mean_target[1:2]) / self.std_target[1:2])) / (x_collocation - ((model.anchor_x - self.mean_target[0:1]) / self.std_target[0:1])))

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
            # da_dP = torch.autograd.grad(
            #     outputs=a_collocation,
            #     inputs=P_collocation,
            #     grad_outputs=torch.ones_like(a_collocation),
            #     create_graph=True
            # )[0]                

            # dx_dP = (self.std_target[0:1] / self.std_input[0:1]) * dx_dP
            # dy_dP = (self.std_target[1:2] / self.std_input[0:1]) * dy_dP
            # da_dP = (self.std_input[1:2] / self.std_input[0:1]) * da_dP

            dx_dP = (self.std_target[0:1] / self.std_input[0:n_anchors]) * dx_dP
            dy_dP = (self.std_target[1:2] / self.std_input[0:n_anchors]) * dy_dP
            # da_dP = (self.std_input[n_anchors:8] / self.std_input[0:n_anchors]) * da_dP

            x_collocation = self.std_target[0:1] * x_collocation + self.mean_target[0:1]
            y_collocation = self.std_target[1:2] * y_collocation + self.mean_target[1:2]
            x_collocation = x_collocation - model.anchor_x  # (N, n_anchors)
            y_collocation = y_collocation - model.anchor_y  # (N, n_anchors)
            distance_2 = torch.clamp(torch.pow(x_collocation, 2) + torch.pow(y_collocation, 2), min=1e-8)  # (N, n_anchors)

            residual_x = dx_dP + torch.einsum('h,nh->nh', model.k, x_collocation)
            residual_y = dy_dP + torch.einsum('h,nh->nh', model.k, y_collocation)
            rss_loss = torch.mean(torch.pow(residual_x, 2)) + torch.mean(torch.pow(residual_y, 2))

            # residual_rss = torch.einsum('nh,nh->nh', x_collocation, dx_dP) + torch.einsum('nh,nh->nh', y_collocation, dy_dP) + torch.einsum('h,nh->nh', model.k, torch.pow(x_collocation, 2)) + torch.einsum('h,nh->nh', model.k, torch.pow(y_collocation, 2))
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
            # residual_azimuth = da_dP - torch.einsum('nh,nh->nh', x_collocation / distance_2, dy_dP) + torch.einsum('nh,nh->nh', y_collocation / distance_2, dx_dP)
            
            # rss_loss = torch.mean(torch.pow(residual_rss, 2))
            # rss_loss = torch.mean(torch.pow(residual_x, 2)) + torch.mean(torch.pow(residual_y, 2))
            # azimuth_loss = torch.mean(torch.pow(residual_azimuth, 2))
            # print('rss_loss: ', rss_loss, ' azimuth_loss: ', azimuth_loss)

            boundary_collocation = self.boundary_collocation_points(n_collocation=self.n_collocation, n_features=inputs.shape[-1], n_anchors=n_anchors,
                                                                    rss_1m=model.rss_1m, mean_input=self.mean_input, std_input=self.std_input,
                                                                    device=inputs.device)
            z_boundary_collocation = model(boundary_collocation)
            x_boundary_collocation = z_boundary_collocation[:,0:1]
            y_boundary_collocation = z_boundary_collocation[:,1:2]
            x_boundary_collocation = self.std_target[0:1] * x_boundary_collocation + self.mean_target[0:1]
            y_boundary_collocation = self.std_target[1:2] * y_boundary_collocation + self.mean_target[1:2]
            x_boundary_collocation = x_boundary_collocation - model.anchor_x
            y_boundary_collocation = y_boundary_collocation - model.anchor_y
            distance_2 = torch.pow(x_boundary_collocation, 2) + torch.pow(y_boundary_collocation, 2)
            z_boundary_target = torch.pow(model.d_0, 2) * torch.ones((self.n_collocation, 1), device=inputs.device, requires_grad=False)
            # print(model.d_0)
            # print(model.rss_1m)
            residual_boundary = distance_2 - z_boundary_target
            boundary_loss = torch.mean(torch.pow(residual_boundary, 2))
            # boundary_loss = self.data_loss_fn(distance_2, z_boundary_target)

            # Compute total loss
            total_loss = (self.lambda_data * data_loss) + (self.lambda_rss * rss_loss) + (self.lambda_azimuth * azimuth_loss) + (self.lambda_bc * boundary_loss)
        
        return total_loss
