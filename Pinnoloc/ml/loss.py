import torch
    

class DistanceLoss(torch.nn.Module):
    def __init__(self, lambda_data, lambda_physics, lambda_bc, n_collocation=1000,
                 mean_input=None, std_input=None, mean_target=None, std_target=None):
        """
        Custom loss function that combines data prediction loss with physics-based loss for RSSI path loss.
        - Input:
            - lambda_data: Weight for data prediction loss.
            - lambda_physics: Weight for physics-based loss.
        """
        if lambda_data < 0.0 or lambda_physics < 0.0 or lambda_bc < 0.0:
            raise ValueError("Lambda weights must be non-negative.")
        super(DistanceLoss, self).__init__()
        self.lambda_data = lambda_data
        self.lambda_physics = lambda_physics
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
    def collocation_points(n_collocation, n_features, device, requires_grad=True):
        # Collocation points for the physics loss representing RSSI with normal distribution
        # P_collocation = torch.randn(n_collocation, 1, device=device, requires_grad=requires_grad)
        # a_collocation = torch.randn(n_collocation, n_features - 1, device=device, requires_grad=False)
        P_collocation = -3.0 + 6.0 * torch.rand(n_collocation, 1, device=device, requires_grad=requires_grad)
        a_collocation = -3.0 + 6.0 * torch.rand(n_collocation, n_features - 1, device=device, requires_grad=False)
        collocation = torch.cat((P_collocation, a_collocation), dim=-1)
        return P_collocation, collocation
    
    @staticmethod
    def boundary_collocation_points(rss_1m, mean_input, std_input, n_collocation, n_features, device):
        # Collocation points for the physics loss representing RSSI with normal distribution
        P0 = (rss_1m - mean_input[:1]) / std_input[:1]
        P0 = torch.einsum('h,nh->nh', P0, torch.ones(n_collocation, 1, device=device, requires_grad=False))
        # a_collocation = torch.randn(n_collocation, n_features - 1, device=device, requires_grad=False)
        a_collocation = -3.0 + 6.0 * torch.rand(n_collocation, n_features - 1, device=device, requires_grad=False)
        boundary_collocation = torch.cat((P0, a_collocation), dim=-1)
        return boundary_collocation

    def forward(self, model, inputs, targets, physics_data=None):
        if self.lambda_physics == 0.0:
            output = model(inputs)
            total_loss = self.data_loss_fn(output, targets)
        
        else:
            output = model(inputs)
            data_loss = self.data_loss_fn(output, targets)  # Data prediction loss

            # Collocation points for the physics loss representing RSSI with normal distribution
            P_collocation, collocation = self.collocation_points(n_collocation=self.n_collocation, n_features=inputs.shape[-1], device=inputs.device, requires_grad=True)
            
            # Physics loss: enforce the differential equation dz/dP = -k * z
            # Compute the derivative dz/dP using autograd,
            # where z represents the distance and P the RSSI in dBm
            z_collocation = model(collocation)
            dz_dP = torch.autograd.grad(
                outputs=z_collocation,
                inputs=P_collocation,
                grad_outputs=torch.ones_like(z_collocation),  # dLoss/dz = 1, dLoss/dP = dLoss/dz * dz/dP
                create_graph=True
            )[0]

            # Differential equation: dz/dP + (ln(10)/(10*path_loss)) * z = 0 if data not standardized, with k = ln(10)/(10*path_loss)
            # dz/dP + k * z = 0
            # z = sigma_z * z* + mu_z, P = sigma_P * P* + mu_P, z* standardized distance, P* standardized RSSI
            # dz/dP = d(sigma_z * z* + mu_z) / d(sigma_P * P* + mu_P) = (sigma_z / sigma_P) * dz*/dP*
            # (sigma_z / sigma_P) * dz*/dP* + k * (sigma_z * z* + mu_z) = 0
            dz_dP = (self.std_target / self.std_input[:1]) * dz_dP
            z_collocation = self.std_target * z_collocation + self.mean_target

            residual = dz_dP + torch.einsum('h,nh->nh', model.k, z_collocation)
            physics_loss = torch.mean(torch.pow(residual, 2))

            if self.lambda_bc == 0.0:
                total_loss = self.lambda_data * data_loss + self.lambda_physics * physics_loss
            else:
                boundary_collocation = self.boundary_collocation_points(rss_1m=model.rss_1m, mean_input=self.mean_input, std_input=self.std_input,
                                                                        n_collocation=self.n_collocation, n_features=inputs.shape[-1], device=inputs.device)
                z_boundary_collocation = model(boundary_collocation)
                z_boundary_target = (torch.ones_like(z_boundary_collocation) - self.mean_target) / self.std_target
                boundary_loss = self.data_loss_fn(z_boundary_collocation, z_boundary_target)

                # Compute total loss
                total_loss = self.lambda_data * data_loss + self.lambda_physics * physics_loss + self.lambda_bc * boundary_loss
        
        return total_loss
    
    def plot(self, model, inputs, targets, physics_data=None):
        # Collocation points for the physics loss representing RSSI with normal distribution
        P_collocation, _ = self.collocation_points(n_collocation=self.n_collocation, device=inputs.device, n_features=inputs.shape[-1], requires_grad=False)
        import matplotlib.pyplot as plt
        P_collocation = P_collocation.cpu().detach()
        k = model.k.cpu().detach()
        inputs = inputs.cpu().detach()
        targets = targets.cpu().detach()

        rss = self.mean_input[:1] + self.std_input[:1] * P_collocation
        exponent = - torch.einsum('h,nh->nh', k, rss - model.rss_1m).cpu().detach()
        # solution = model.boundary_condition * torch.exp(exponent).cpu().detach()
        solution = torch.exp(exponent).cpu().detach()

        z_collocation = model(P_collocation).cpu().detach()
        preds = model(inputs).cpu().detach()

        P_collocation = self.std_input[:1] * P_collocation + self.mean_input[:1]
        inputs = self.std_input[:1] * inputs + self.mean_input[:1]
        targets = self.std_target * targets + self.mean_target
        preds = self.std_target * preds + self.mean_target
        z_collocation = self.std_target * z_collocation + self.mean_target

        plt.scatter(P_collocation.numpy(), solution.numpy(), label='Equation')
        plt.scatter(P_collocation.numpy(), z_collocation.numpy(), label='Predicted Collocation')
        plt.scatter(inputs.numpy(), targets.numpy(), label='True Data')
        plt.scatter(inputs.numpy(), preds.numpy(), label='Predicted Data')
        plt.xlabel('RSSI P (dBm)')
        plt.ylabel('Distance z (m)')
        plt.legend()
        plt.show()


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
    def collocation_points(n_collocation, n_features, device, requires_grad=True):
        # Collocation points for the physics loss representing RSSI with normal distribution
        P_collocation = torch.randn(n_collocation, 1, device=device, requires_grad=requires_grad)
        other_collocation = torch.randn(n_collocation, n_features - 1, device=device, requires_grad=False)
        # P_collocation = -3.0 + 6.0 * torch.rand(n_collocation, 1, device=device, requires_grad=requires_grad)
        # other_collocation = -3.0 + 6.0 * torch.rand(n_collocation, n_features - 1, device=device, requires_grad=False)
        collocation = torch.cat((P_collocation, other_collocation), dim=-1)
        return P_collocation, collocation
    
    @staticmethod
    def boundary_collocation_points(rss_1m, mean_input, std_input, n_collocation, n_features, device):
        # Collocation points for the physics loss representing RSSI with normal distribution
        P0 = (rss_1m - mean_input[0:1]) / std_input[0:1]
        P0 = torch.einsum('h,nh->nh', P0, torch.ones(n_collocation, 1, device=device, requires_grad=False))
        other_collocation = torch.randn(n_collocation, n_features - 1, device=device, requires_grad=False)
        # other_collocation = -3.0 + 6.0 * torch.rand(n_collocation, n_features - 1, device=device, requires_grad=False)
        boundary_collocation = torch.cat((P0, other_collocation), dim=-1)
        return boundary_collocation

    def forward(self, model, inputs, targets, physics_data=None):
        if self.lambda_rss == 0.0 and self.lambda_azimuth == 0.0 and self.lambda_bc == 0.0:
            output = model(inputs)
            total_loss = self.data_loss_fn(output, targets)
        
        else:
            output = model(inputs)
            data_loss = self.data_loss_fn(output, targets)  # Data prediction loss

            # Collocation points for the physics loss representing RSSI with normal distribution
            P_collocation, collocation = self.collocation_points(n_collocation=self.n_collocation, n_features=inputs.shape[-1], device=inputs.device, requires_grad=True)
            
            z_collocation = model(collocation)
            x_collocation = z_collocation[:,0:1]
            y_collocation = z_collocation[:,1:2]
            a_collocation = torch.atan(y_collocation - ((model.anchor_y - self.mean_target[1:2]) / self.std_target[1:2]) / x_collocation - ((model.anchor_x - self.mean_target[0:1]) / self.std_target[0:1]))
            # a_collocation = torch.atan2(y_collocation - ((model.anchor_y - self.mean_target[1:2]) / self.std_target[1:2]), x_collocation - ((model.anchor_x - self.mean_target[0:1]) / self.std_target[0:1]))

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
            da_dP = torch.autograd.grad(
                outputs=a_collocation,
                inputs=P_collocation,
                grad_outputs=torch.ones_like(a_collocation),
                create_graph=True
            )[0]

            dx_dP = (self.std_target[0:1] / self.std_input[0:1]) * dx_dP
            dy_dP = (self.std_target[1:2] / self.std_input[0:1]) * dy_dP
            da_dP = (self.std_input[1:2] / self.std_input[0:1]) * da_dP

            x_collocation = self.std_target[0:1] * x_collocation + self.mean_target[0:1]
            y_collocation = self.std_target[1:2] * y_collocation + self.mean_target[1:2]
            x_collocation = x_collocation - model.anchor_x
            y_collocation = y_collocation - model.anchor_y
            distance_2 = torch.pow(x_collocation, 2) + torch.pow(y_collocation, 2)

            residual_x = dx_dP + torch.einsum('h,nh->nh', model.k, x_collocation)
            residual_y = dy_dP + torch.einsum('h,nh->nh', model.k, y_collocation)
            # residual_rss = torch.einsum('nh,nh->nh', x_collocation, dx_dP) + torch.einsum('nh,nh->nh', y_collocation, dy_dP) + torch.einsum('h,nh->nh', model.k, torch.pow(x_collocation, 2)) + torch.einsum('h,nh->nh', model.k, torch.pow(y_collocation, 2))
            residual_azimuth = da_dP - torch.einsum('nh,nh->nh', x_collocation / distance_2, dy_dP) + torch.einsum('nh,nh->nh', y_collocation / distance_2, dx_dP)
            
            # rss_loss = torch.mean(torch.pow(residual_rss, 2))
            rss_loss = torch.mean(torch.pow(residual_x, 2)) + torch.mean(torch.pow(residual_y, 2))
            azimuth_loss = torch.mean(torch.pow(residual_azimuth, 2))

            boundary_collocation = self.boundary_collocation_points(rss_1m=model.rss_1m, mean_input=self.mean_input, std_input=self.std_input,
                                                                    n_collocation=self.n_collocation, n_features=inputs.shape[-1], device=inputs.device)
            z_boundary_collocation = model(boundary_collocation)
            x_boundary_collocation = z_boundary_collocation[:,0:1]
            y_boundary_collocation = z_boundary_collocation[:,1:2]
            x_boundary_collocation = self.std_target[0:1] * x_boundary_collocation + self.mean_target[0:1]
            y_boundary_collocation = self.std_target[1:2] * y_boundary_collocation + self.mean_target[1:2]
            x_boundary_collocation = x_boundary_collocation - model.anchor_x
            y_boundary_collocation = y_boundary_collocation - model.anchor_y
            distance_2 = torch.pow(x_boundary_collocation, 2) + torch.pow(y_boundary_collocation, 2)
            z_boundary_target = torch.ones((self.n_collocation, 1), device=inputs.device, requires_grad=False)
            boundary_loss = self.data_loss_fn(distance_2, z_boundary_target)

            # Compute total loss
            total_loss = self.lambda_data * data_loss + self.lambda_rss * rss_loss + self.lambda_azimuth * azimuth_loss + self.lambda_bc * boundary_loss
        
        return total_loss


class DistanceLossIMG(torch.nn.Module):
    def __init__(self, lambda_data, lambda_physics, lambda_bc, n_collocation=1000,
                 mean_input=None, std_input=None, mean_target=None, std_target=None):
        """
        Custom loss function that combines data prediction loss with physics-based loss for RSSI path loss.
        - Input:
            - lambda_data: Weight for data prediction loss.
            - lambda_physics: Weight for physics-based loss.
        """
        if lambda_data < 0.0 or lambda_physics < 0.0 or lambda_bc < 0.0:
            raise ValueError("Lambda weights must be non-negative.")
        super(DistanceLossIMG, self).__init__()
        self.lambda_data = lambda_data
        self.lambda_physics = lambda_physics
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
    def collocation_points(n_collocation, input_channels, input_height, input_width, device, requires_grad=True):
        # Collocation points for the physics loss representing RSSI with normal distribution
        P_collocation = torch.randn(n_collocation, input_channels, input_height, 2,
                                    device=device, requires_grad=requires_grad)
        a_collocation = torch.randn(n_collocation, input_channels, input_height, input_width - 2,
                                    device=device, requires_grad=False)
        collocation = torch.cat((P_collocation, a_collocation), dim=-1)
        return P_collocation, collocation
    
    @staticmethod
    def boundary_collocation_points(rss_1m, mean_input, std_input, n_collocation, input_channels, input_height, input_width, device):
        # Collocation points for the physics loss representing RSSI with normal distribution
        P0 = (rss_1m - mean_input[:2]) / std_input[:2]
        P0 = torch.einsum('hw,nchw->nchw', P0, torch.ones(n_collocation, input_channels, input_height, 2, device=device, requires_grad=False))
        a_collocation = torch.randn(n_collocation, input_channels, input_height, input_width - 2, device=device, requires_grad=False)
        boundary_collocation = torch.cat((P0, a_collocation), dim=-1)
        return boundary_collocation

    def forward(self, model, inputs, targets, physics_data=None):
        if self.lambda_physics == 0.0:
            output = model(inputs)
            total_loss = self.data_loss_fn(output, targets)
        
        else:
            output = model(inputs)
            data_loss = self.data_loss_fn(output, targets)  # Data prediction loss

            # Collocation points for the physics loss representing RSSI with normal distribution
            P_collocation, collocation = self.collocation_points(n_collocation=self.n_collocation,
                                                                input_channels=inputs.shape[1], input_height=inputs.shape[2], input_width=inputs.shape[3],
                                                                device=inputs.device, requires_grad=True)
            
            # Physics loss: enforce the differential equation dz/dP = -k * z
            # Compute the derivative dz/dP using autograd,
            # where z represents the distance and P the RSSI in dBm
            z_collocation = model(collocation)
            print(z_collocation.shape)
            print(P_collocation.shape)
            print(collocation.shape)
            
            dz_dP_00 = torch.autograd.grad(
                outputs=z_collocation[:,0:1],
                inputs=P_collocation[:,:,0:1,0:1],
                grad_outputs=torch.ones_like(z_collocation[:,0:1]),  # dLoss/dz = 1, dLoss/dP = dLoss/dz * dz/dP
                create_graph=True
            )[0]

            dz_dP_01 = torch.autograd.grad(
                outputs=z_collocation[:,0:1],
                inputs=P_collocation[:,:,0:1,1:2],
                grad_outputs=torch.ones_like(z_collocation[:,0:1]),  # dLoss/dz = 1, dLoss/dP = dLoss/dz * dz/dP
                create_graph=True
            )[0]

            dz_dP_10 = torch.autograd.grad(
                outputs=z_collocation[:,1:2],
                inputs=P_collocation[:,:,1:2,0:1],
                grad_outputs=torch.ones_like(z_collocation[:,0:1]),  # dLoss/dz = 1, dLoss/dP = dLoss/dz * dz/dP
                create_graph=True
            )[0]

            dz_dP_11 = torch.autograd.grad(
                outputs=z_collocation[:,1:2],
                inputs=P_collocation[:,:,1:2,1:2],
                grad_outputs=torch.ones_like(z_collocation[:,1:2]),  # dLoss/dz = 1, dLoss/dP = dLoss/dz * dz/dP
                create_graph=True
            )[0]

            dz_dP_20 = torch.autograd.grad(
                outputs=z_collocation[:,2:3],
                inputs=P_collocation[:,:,2:3,0:1],
                grad_outputs=torch.ones_like(z_collocation[:,2:3]),  # dLoss/dz = 1, dLoss/dP = dLoss/dz * dz/dP
                create_graph=True
            )[0]

            dz_dP_21 = torch.autograd.grad(
                outputs=z_collocation[:,2:3],
                inputs=P_collocation[:,:,2:3,1:2],
                grad_outputs=torch.ones_like(z_collocation[:,2:3]),  # dLoss/dz = 1, dLoss/dP = dLoss/dz * dz/dP
                create_graph=True
            )[0]

            dz_dP_30 = torch.autograd.grad(
                outputs=z_collocation[:,3:4],
                inputs=P_collocation[:,:,3:4,0:1],
                grad_outputs=torch.ones_like(z_collocation[:,3:4]),  # dLoss/dz = 1, dLoss/dP = dLoss/dz * dz/dP
                create_graph=True
            )[0]

            dz_dP_31 = torch.autograd.grad(
                outputs=z_collocation[:,3:4],
                inputs=P_collocation[:,:,3:4,1:2],
                grad_outputs=torch.ones_like(z_collocation[:,3:4]),  # dLoss/dz = 1, dLoss/dP = dLoss/dz * dz/dP
                create_graph=True
            )[0]

            dz_dP = torch.as_tensor([[dz_dP_00, dz_dP_01],
                                     [dz_dP_10, dz_dP_11],
                                     [dz_dP_20, dz_dP_21],
                                     [dz_dP_30, dz_dP_31]])


            # Differential equation: dz/dP + (ln(10)/(10*path_loss)) * z = 0 if data not standardized, with k = ln(10)/(10*path_loss)
            # dz/dP + k * z = 0
            # z = sigma_z * z* + mu_z, P = sigma_P * P* + mu_P, z* standardized distance, P* standardized RSSI
            # dz/dP = d(sigma_z * z* + mu_z) / d(sigma_P * P* + mu_P) = (sigma_z / sigma_P) * dz*/dP*
            # (sigma_z / sigma_P) * dz*/dP* + k * (sigma_z * z* + mu_z) = 0
            dz_dP = (self.std_target / self.std_input[:2]) * dz_dP
            z_collocation = self.std_target * z_collocation + self.mean_target

            residual = dz_dP + torch.einsum('hw,nw->nhw', model.k, z_collocation)
            physics_loss = torch.mean(torch.pow(residual, 2))

            if self.lambda_bc == 0.0:
                total_loss = self.lambda_data * data_loss + self.lambda_physics * physics_loss
            else:
                boundary_collocation = self.boundary_collocation_points(rss_1m=model.rss_1m, mean_input=self.mean_input, std_input=self.std_input,
                                                                        n_collocation=self.n_collocation, n_features=inputs.shape[-1], device=inputs.device)
                z_boundary_collocation = model(boundary_collocation)
                z_boundary_max = (torch.ones_like(z_boundary_collocation) - self.mean_target) / self.std_target
                azimuth = torch.atan2(boundary_collocation[:, 2:3], boundary_collocation[:, 1:2])
                elevation = torch.atan2(boundary_collocation[:, 4:5], boundary_collocation[:, 3:4])
                # Gaussian function for boundary condition with azimuth and elevation
                z_boundary_gauss = torch.exp(-((torch.pow(azimuth, 2) + torch.pow(elevation, 2)) / (2 * (model.sigma ** 2))))
                z_boundary_target = z_boundary_max * z_boundary_gauss

                boundary_loss = self.data_loss_fn(z_boundary_collocation, z_boundary_target)

                # Compute total loss
                total_loss = self.lambda_data * data_loss + self.lambda_physics * physics_loss + self.lambda_bc * boundary_loss
        
        return total_loss
    
    def plot(self, model, inputs, targets, physics_data=None):
        # Collocation points for the physics loss representing RSSI with normal distribution
        P_collocation, _ = self.collocation_points(n_collocation=self.n_collocation, device=inputs.device, n_features=inputs.shape[-1], requires_grad=False)
        import matplotlib.pyplot as plt
        P_collocation = P_collocation.cpu().detach()
        k = model.k.cpu().detach()
        inputs = inputs.cpu().detach()
        targets = targets.cpu().detach()

        rss = self.mean_input[:1] + self.std_input[:1] * P_collocation
        exponent = - torch.einsum('z,nz->nz', k, rss - model.rss_1m).cpu().detach()
        # solution = model.boundary_condition * torch.exp(exponent).cpu().detach()
        solution = torch.exp(exponent).cpu().detach()

        z_collocation = model(P_collocation).cpu().detach()
        preds = model(inputs).cpu().detach()

        P_collocation = self.std_input[:1] * P_collocation + self.mean_input[:1]
        inputs = self.std_input[:1] * inputs + self.mean_input[:1]
        targets = self.std_target * targets + self.mean_target
        preds = self.std_target * preds + self.mean_target
        z_collocation = self.std_target * z_collocation + self.mean_target

        plt.scatter(P_collocation.numpy(), solution.numpy(), label='Equation')
        plt.scatter(P_collocation.numpy(), z_collocation.numpy(), label='Predicted Collocation')
        plt.scatter(inputs.numpy(), targets.numpy(), label='True Data')
        plt.scatter(inputs.numpy(), preds.numpy(), label='Predicted Data')
        plt.xlabel('RSSI P (dBm)')
        plt.ylabel('Distance z (m)')
        plt.legend()
        plt.show()