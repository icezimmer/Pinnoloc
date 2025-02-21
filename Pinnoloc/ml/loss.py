import torch


# class PhysicsInformedLoss(torch.nn.Module):
#     def __init__(self, lambda_rss=0.1, lambda_aoa=0.1, A_max=10):
#         super(PhysicsInformedLoss, self).__init__()
#         self.lambda_rss = lambda_rss
#         self.lambda_aoa = lambda_aoa
#         self.A_max = A_max
#         self.mse_loss = torch.nn.MSELoss()

#     def forward(self, pred_theta, true_theta, rssi, aoa, distance, heading, anchor_angles):
#         mse_loss = self.mse_loss(pred_theta, true_theta)  # Heading prediction loss

#         # Blocking detection (1 if blocked, 0 otherwise)
#         block_condition = (torch.abs(heading - anchor_angles) > torch.pi / 2).float()

#         # RSSI loss
#         path_loss_pred = -10 * torch.log10(distance) - block_condition * self.A_max
#         rss_loss = torch.mean((rssi - path_loss_pred) ** 2)

#         # AoA loss
#         d_ant = 0.05  # Antenna spacing (50mm)
#         lambda_signal = 0.125  # BLE 2.4 GHz wavelength
#         theta_body_error = torch.randn_like(pred_theta) * block_condition * 5  # Add AoA error if blocked
#         phi_pred = (2 * torch.pi * d_ant / lambda_signal) * torch.sin(pred_theta + theta_body_error)
#         aoa_loss = torch.mean((aoa - phi_pred) ** 2)

#         return mse_loss + self.lambda_rss * rss_loss + self.lambda_aoa * aoa_loss


# class PhysicsInformedLoss(torch.nn.Module):
#     def __init__(self, lambda_data, lambda_rss):
#         super(PhysicsInformedLoss, self).__init__()
#         self.lambda_data = lambda_data
#         self.lambda_rss = lambda_rss
#         self.data_loss = torch.nn.CrossEntropyLoss()
#         self.rss_loss_fn = torch.nn.MSELoss()

#     def forward(self, pred_data, true_data, physics_data=None):
#         if physics_data is None:
#             total_loss = self.data_loss(pred_data, true_data)
        
#         else:
#             distance = physics_data[:, 0]
#             rss_obs_1 = physics_data[:, 1]
#             rss_obs_2 = physics_data[:, 2]
#             rss_1m = -40
#             alpha = 2.5

#             data_loss = self.data_loss(pred_data, true_data)  # Data prediction loss

#             # RSSI path loss
#             rss_est = rss_1m - 10 * alpha * torch.log10(distance)
#             # Compute RSSI loss
#             rss_loss_1 = torch.mean((rss_obs_1 - rss_est) ** 2)
#             rss_loss_2 = torch.mean((rss_obs_2 - rss_est) ** 2)

#             # Compute total loss
#             total_loss = self.lambda_data * data_loss + self.lambda_rss * (rss_loss_1 + rss_loss_2)
        
#         return total_loss

# class PhysicsInformedLoss(torch.nn.Module):
#     def __init__(self, lambda_data, lambda_physics):
#         super(PhysicsInformedLoss, self).__init__()
#         self.lambda_data = lambda_data
#         self.lambda_physics = lambda_physics
#         self.data_loss_fn = torch.nn.CrossEntropyLoss()
#         self.physics_loss_fn = torch.nn.MSELoss()

#     @staticmethod
#     def compute_AoD_soft(heading_logits, AoD):
#         """
#         Compute differentiable azimuth angle of departure (AoD_final) using softmax-weighted transformations
#         and vectorized sin-cos interpolation.

#         Args:
#         - heading_logits: [batch_size, num_classes] Raw output logits for heading (before softmax).
#         - AoD: [batch_size] Initial azimuth angles of departure (radians).

#         Returns:
#         - AoD_final: [batch_size] Adjusted azimuth angles based on softmax weighting.
#         """
#         # Compute softmax probabilities over the 4 classes (headings)
#         probabilities = torch.softmax(heading_logits, dim=-1)  # Shape: [batch_size, num_classes]

#         # Define transformation rules for each class (working in vector space)
#         AoD_east  = -AoD                      # Heading = 0 (East)
#         AoD_north = (torch.pi/2) - AoD        # Heading = 1 (North)
#         AoD_south = (-torch.pi/2) - AoD       # Heading = 2 (South)
#         AoD_west  = torch.pi - AoD            # Heading = 3 (West)

#         # Convert angles to (cos, sin) representations
#         cos_east, sin_east  = torch.cos(AoD_east),  torch.sin(AoD_east)
#         cos_north, sin_north = torch.cos(AoD_north), torch.sin(AoD_north)
#         cos_south, sin_south = torch.cos(AoD_south), torch.sin(AoD_south)
#         cos_west, sin_west  = torch.cos(AoD_west),  torch.sin(AoD_west)

#         # Compute weighted sum in vector space
#         AoD_cos = (probabilities[:, 0] * cos_east +
#                 probabilities[:, 1] * cos_north +
#                 probabilities[:, 2] * cos_south +
#                 probabilities[:, 3] * cos_west)

#         AoD_sin = (probabilities[:, 0] * sin_east +
#                 probabilities[:, 1] * sin_north +
#                 probabilities[:, 2] * sin_south +
#                 probabilities[:, 3] * sin_west)

#         # Convert back to angle using atan2
#         AoD_final = torch.atan2(AoD_sin, AoD_cos)  # Shape: [batch_size]

#         return AoD_final

    
#     # def compute_AoD_soft(heading_logits, AoD_cos_sin):
#     #     """
#     #     Compute differentiable azimuth angle of departure (AoD_final) using softmax-weighted transformations
#     #     while working directly with cos(AoD) and sin(AoD) vectors.

#     #     Args:
#     #     - heading_logits: [batch_size, num_classes] Raw output logits for heading (before softmax).
#     #     - AoD_cos_sin: [batch_size, num_anchors, 2] Tensor containing [cos(AoD), sin(AoD)].

#     #     Returns:
#     #     - AoD_final: [batch_size, num_anchors] Adjusted azimuth angles based on softmax weighting.
#     #     """
#     #     # Compute softmax probabilities over the 4 classes (headings)
#     #     probabilities = torch.softmax(heading_logits, dim=-1)  # Shape: [batch_size, num_classes]

#     #     # Extract cos(AoD) and sin(AoD)
#     #     AoD_cos, AoD_sin = AoD_cos_sin[..., 0], AoD_cos_sin[..., 1]  # Shape: [batch_size, num_anchors]

#     #     # Define transformation rules in vector form
#     #     cos_east, sin_east  = -AoD_cos, -AoD_sin                      # East: (cos, sin) -> (-cos, -sin)
#     #     cos_north, sin_north = -AoD_sin, AoD_cos                      # North: (cos, sin) -> (-sin, cos)
#     #     cos_south, sin_south = AoD_sin, -AoD_cos                      # South: (cos, sin) -> (sin, -cos)
#     #     cos_west, sin_west  = -AoD_cos, AoD_sin                       # West: (cos, sin) -> (-cos, sin)

#     #     # Compute weighted sum in vector space
#     #     AoD_cos_final = (probabilities[:, 0].unsqueeze(-1) * cos_east +
#     #                     probabilities[:, 1].unsqueeze(-1) * cos_north +
#     #                     probabilities[:, 2].unsqueeze(-1) * cos_south +
#     #                     probabilities[:, 3].unsqueeze(-1) * cos_west)

#     #     AoD_sin_final = (probabilities[:, 0].unsqueeze(-1) * sin_east +
#     #                     probabilities[:, 1].unsqueeze(-1) * sin_north +
#     #                     probabilities[:, 2].unsqueeze(-1) * sin_south +
#     #                     probabilities[:, 3].unsqueeze(-1) * sin_west)

#     #     # Convert back to angle using atan2
#     #     AoD_final = torch.atan2(AoD_sin_final, AoD_cos_final)  # Shape: [batch_size, num_anchors]

#     #     return AoD_final

#     def forward(self, pred_data, true_data, physics_data=None):
#         if physics_data is None:
#             total_loss = self.data_loss_fn(pred_data, true_data)
        
#         else:
#             data_loss = self.data_loss_fn(pred_data, true_data)  # Data prediction loss

#             distance = physics_data[:, 0]
#             angle_of_departure = self.compute_AoD_soft(heading_logits=pred_data, AoD=physics_data[:, 1])
#             rss_obs_1 = physics_data[:, 2]
#             rss_obs_2 = physics_data[:, 3]
#             # angle_of_departure = physics_data[:, 1:3]
#             # angle_of_departure = self.compute_AoD_soft(pred_data, angle_of_departure)
#             # rss_obs_1 = physics_data[:, 3]
#             # rss_obs_2 = physics_data[:, 4]

#             rss_1m = -40  # RSSI at 1m
#             alpha = 2.5  # Path loss exponent
#             A_max = 10  # Maximum path loss

#             # RSSI path loss with blockage, assuming [-pi/2, pi/2] no interference
#             block_condition = (torch.abs(angle_of_departure) > torch.pi / 2).float()
#             rss_est = rss_1m - 10 * alpha * torch.log10(distance) - block_condition * A_max

#             # Compute RSSI loss
#             rss_loss_1 = self.physics_loss_fn(rss_obs_1, rss_est)
#             rss_loss_2 = self.physics_loss_fn(rss_obs_2, rss_est)
#             # print(rss_obs_1.shape, rss_obs_2.shape, distance.shape, angle_of_departure.shape, rss_est.shape)

#             # Compute total loss
#             total_loss = self.lambda_data * data_loss + self.lambda_physics * (rss_loss_1 + rss_loss_2)
        
#         return total_loss
    

class HeadingLoss(torch.nn.Module):
    def __init__(self, lambda_data, lambda_physics, A_max):
        """
        Custom loss function that combines data prediction loss with physics-based loss for RSSI path loss.
        - Input:
            - lambda_data: Weight for data prediction loss.
            - lambda_physics: Weight for physics-based loss.
            - A_max: Maximum path loss due to blockage.
        """
        super(HeadingLoss, self).__init__()
        self.lambda_data = lambda_data
        self.lambda_physics = lambda_physics
        self.data_loss_fn = torch.nn.CrossEntropyLoss()
        self.physics_loss_fn = torch.nn.MSELoss()
        self.A_max = A_max

    @staticmethod
    def compute_AoD_soft(heading_logits, AoD):
        """
        Compute differentiable azimuth angle of departure (AoD_final) using softmax-weighted transformations
        and vectorized sin-cos interpolation.

        Args:
        - heading_logits: [batch_size, num_classes] Raw output logits for heading (before softmax).
        - AoD: [batch_size] Initial azimuth angles of departure (radians).

        Returns:
        - AoD_final: [batch_size] Adjusted azimuth angles based on softmax weighting.
        """
        # Compute softmax probabilities over the 4 classes (headings)
        probabilities = torch.softmax(heading_logits, dim=-1)  # Shape: [batch_size, num_classes]

        # Define transformation rules for each class (working in vector space)
        AoD_east  = -AoD                      # Heading = 0 (East)
        AoD_north = (torch.pi/2) - AoD        # Heading = 1 (North)
        AoD_south = (-torch.pi/2) - AoD       # Heading = 2 (South)
        AoD_west  = torch.pi - AoD            # Heading = 3 (West)

        # Convert angles to (cos, sin) representations
        cos_east, sin_east  = torch.cos(AoD_east),  torch.sin(AoD_east)
        cos_north, sin_north = torch.cos(AoD_north), torch.sin(AoD_north)
        cos_south, sin_south = torch.cos(AoD_south), torch.sin(AoD_south)
        cos_west, sin_west  = torch.cos(AoD_west),  torch.sin(AoD_west)

        # Compute weighted sum in vector space
        AoD_cos = (probabilities[:, 0] * cos_east +
                probabilities[:, 1] * cos_north +
                probabilities[:, 2] * cos_south +
                probabilities[:, 3] * cos_west)

        AoD_sin = (probabilities[:, 0] * sin_east +
                probabilities[:, 1] * sin_north +
                probabilities[:, 2] * sin_south +
                probabilities[:, 3] * sin_west)

        # Convert back to angle using atan2
        AoD_final = torch.atan2(AoD_sin, AoD_cos)  # Shape: [batch_size]

        return AoD_final

    def forward(self, pred_data, true_data, physics_data=None):
        if physics_data is None:
            total_loss = self.data_loss_fn(pred_data, true_data)
        
        else:
            data_loss = self.data_loss_fn(pred_data, true_data)  # Data prediction loss

            distance = physics_data[:, 0]
            rss_obs = physics_data[:, 1]
            angle_of_departure = self.compute_AoD_soft(heading_logits=pred_data, AoD=physics_data[:, 2])
            rss_1m = physics_data[:, 3]
            alpha = physics_data[:, 4]

            # RSSI path loss with blockage, assuming [-pi/2, pi/2] no interference
            block_condition = (torch.abs(angle_of_departure) > torch.pi / 2).float()
            block_factor = ((torch.abs(angle_of_departure) - (torch.pi / 2)) / (torch.pi / 2)) * block_condition
            rss_est = rss_1m - 10 * alpha * torch.log10(distance) - block_factor * self.A_max
            # print("orginal: ", physics_data[:, 2])
            # print("adjusted: ", angle_of_departure)
            # print("block_condition: ", block_condition)
            # print("block_factor: ", block_factor)

            # Compute RSSI loss
            rss_loss = self.physics_loss_fn(rss_obs, rss_est)
            # print(rss_obs_1.shape, rss_obs_2.shape, distance.shape, angle_of_departure.shape, rss_est.shape)

            # Compute total loss
            total_loss = self.lambda_data * data_loss + self.lambda_physics * rss_loss
        
        return total_loss
    

# class DistanceLoss(torch.nn.Module):
#     def __init__(self, lambda_data, lambda_physics):
#         """
#         Custom loss function that combines data prediction loss with physics-based loss for RSSI path loss.
#         - Input:
#             - lambda_data: Weight for data prediction loss.
#             - lambda_physics: Weight for physics-based loss.
#         """
#         super(DistanceLoss, self).__init__()
#         self.lambda_data = lambda_data
#         self.lambda_physics = lambda_physics
#         self.data_loss_fn = torch.nn.MSELoss()
#         self.physics_loss_fn = torch.nn.MSELoss()

#     def forward(self, pred_data, true_data, physics_data=None):
#         if physics_data is None:
#             total_loss = self.data_loss_fn(pred_data, true_data)
        
#         else:
#             data_loss = self.data_loss_fn(pred_data, true_data)  # Data prediction loss

#             distance_mean = physics_data[:, 0]
#             distance_std = physics_data[:, 1]
#             # Reconstruct distance with original mean and std avoiding negative values
#             distance = distance_mean + distance_std * pred_data[:, 0]
#             distance = torch.max(distance, torch.zeros_like(distance) + 0.01)  # Avoid negative values

#             rss_obs = physics_data[:, 2]
#             rss_1m = physics_data[:, 3]
#             alpha = physics_data[:, 4]


#             # RSSI path loss with blockage, assuming [-pi/2, pi/2] no interference
#             rss_est = rss_1m - 10 * alpha * torch.log10(distance)
#             # print("orginal: ", physics_data[:, 2])
#             # print("adjusted: ", angle_of_departure)
#             # print("block_condition: ", block_condition)
#             # print("block_factor: ", block_factor)

#             # Compute RSSI loss
#             rss_loss = self.physics_loss_fn(rss_obs, rss_est)
#             # print(rss_obs_1.shape, rss_obs_2.shape, distance.shape, angle_of_departure.shape, rss_est.shape)

#             # Compute total loss
#             total_loss = self.lambda_data * data_loss + self.lambda_physics * rss_loss
        
#         return total_loss
    

class DistanceLoss(torch.nn.Module):
    def __init__(self, lambda_data, lambda_physics, mean_input=0.0, std_input=1.0, mean_target=0.0, std_target=1.0):
        """
        Custom loss function that combines data prediction loss with physics-based loss for RSSI path loss.
        - Input:
            - lambda_data: Weight for data prediction loss.
            - lambda_physics: Weight for physics-based loss.
        """
        if lambda_data < 0 or lambda_physics < 0:
            raise ValueError("Lambda weights must be non-negative.")
        super(DistanceLoss, self).__init__()
        self.lambda_data = lambda_data
        self.lambda_physics = lambda_physics
        self.data_loss_fn = torch.nn.MSELoss()
        self.mean_input = mean_input
        self.std_input = std_input
        self.mean_target = mean_target
        self.std_target = std_target

    @staticmethod
    def collocation_points(device, n_features, requires_grad=True):
        # Collocation points for the physics loss representing RSSI with normal distribution
        # P_collocation = torch.linspace(-3, 3, 1000, device=device, requires_grad=requires_grad).view(-1, 1)
        P_collocation = torch.randn(10000, 1, device=device, requires_grad=requires_grad)
        a_collocation = torch.randn(10000, n_features - 1, device=device, requires_grad=False)
        collocation = torch.cat((P_collocation, a_collocation), dim=1)
        return P_collocation, collocation
    
    @staticmethod
    def collocation_boundary_points(device, n_features):
        # Collocation points for the physics loss representing RSSI with normal distribution
        P0 = torch.zeros(10000, 1, device=device, requires_grad=False)
        a_collocation = torch.randn(10000, n_features - 1, device=device, requires_grad=False)
        boundary_collocation = torch.cat((P0, a_collocation), dim=1)
        return boundary_collocation

    def forward(self, model, inputs, targets, physics_data=None):
        if self.lambda_physics == 0:
            output = model(inputs)
            total_loss = self.data_loss_fn(output, targets)
        
        else:
            output = model(inputs)
            data_loss = self.data_loss_fn(output, targets)  # Data prediction loss

            # Collocation points for the physics loss representing RSSI with normal distribution
            P_collocation, collocation = self.collocation_points(device=inputs.device, n_features=inputs.shape[-1], requires_grad=True)
            
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
            
            # Differential equation: dz/dP + (ln(10)/(10*alpha)) * z = 0 if data not standardized
            # z = sigma_z * z* + mu_z, P = sigma_P * P* + mu_P, z* standardized distance, P* standardized RSSI
            # dz/dP = d(sigma_z * z* + mu_z) / d(sigma_P * P* + mu_P) = (sigma_z / sigma_P) * dz*/dP*
            # alpha = physics_data[0, 0]
            # z_collocation = self.std_target * z_collocation + self.mean_target
            # dz_dP = (self.std_target / self.std_input) * dz_dP
            # k = torch.log(torch.tensor(10.0)) / (10 * model.alpha)
            # residual = dz_dP + k * z_pred_collocation
            residual = dz_dP + torch.einsum('f,nf->nf', model.path_loss, z_collocation)
            physics_loss = torch.mean(torch.pow(residual, 2))

            # P0 = torch.tensor(-50.0, device=inputs.device).view(-1,1)
            # z0 = torch.tensor(1.0, device=inputs.device).view(-1,1)

            # ic_loss = self.data_loss_fn(model(P0), z0)

            boundary_collocation = self.collocation_boundary_points(device=inputs.device, n_features=inputs.shape[-1])
            z_boundary_collocation = model(boundary_collocation)
            target_boundary = model.boundary_condition * torch.empty_like(z_boundary_collocation).fill_(1.0)
            boundary_loss = self.data_loss_fn(z_boundary_collocation, target_boundary)
            
            # print(f"Data loss: {data_loss.item()}, Physics loss: {physics_loss.item()}, IC loss: {ic_loss.item()}")

            # Compute total loss
            total_loss = self.lambda_data * data_loss + self.lambda_physics * physics_loss + 100 * boundary_loss # + 100 * ic_loss
            # print(f"Data loss: {data_loss.item()}, Physics loss: {physics_loss.item()}")
        
        return total_loss
    
    def plot(self, model, inputs, targets, physics_data=None):
        # Collocation points for the physics loss representing RSSI with normal distribution
        P_collocation, _ = self.collocation_points(device=inputs.device, n_features=inputs.shape[-1], requires_grad=False)
        import matplotlib.pyplot as plt
        P_collocation = P_collocation.cpu().detach()
        path_loss = model.path_loss.cpu().detach()
        inputs = inputs.cpu().detach()
        targets = targets.cpu().detach()

        exponent = - torch.einsum('f,nf->nf', path_loss, P_collocation).cpu().detach()
        solution = model.boundary_condition * torch.exp(exponent).cpu().detach()

        z_collocation = model(P_collocation).cpu().detach()
        preds = model(inputs).cpu().detach()

        plt.scatter(P_collocation.numpy(), solution.numpy(), label='Equation')
        plt.scatter(P_collocation.numpy(), z_collocation.numpy(), label='Predicted Collocation')
        plt.scatter(inputs.numpy(), targets.numpy(), label='True Data')
        plt.scatter(inputs.numpy(), preds.numpy(), label='Predicted Data')
        plt.xlabel('RSSI P (dBm)')
        plt.ylabel('Distance z (m)')
        plt.legend()
        plt.show()