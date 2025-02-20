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

    def plot(self, model, input, target, physics_data=None):
        # Collocation points for the physics loss representing RSSI with normal distribution
        P_collocation = self.collocation_points(device=input.device, requires_grad=False)
        # x_pred_collocation = model(P_collocation)
        # dx_dP = torch.autograd.grad(
        #     outputs=x_pred_collocation,
        #     inputs=P_collocation,
        #     grad_outputs=torch.ones_like(x_pred_collocation),  # dLoss/dx = 1, dLoss/dP = dLoss/dx * dx/dP
        #     create_graph=False
        # )[0]
        # approx = - torch.einsum('f,bf->bf', model.path_loss, x_pred_collocation)
        # Scatter plot
        import matplotlib.pyplot as plt
        plt.scatter(P_collocation.cpu().detach().numpy(), model(P_collocation).cpu().detach().numpy(), label='Predicted Collocation')
        plt.scatter(input.cpu().detach().numpy(), target.cpu().detach().numpy(), label='True Data')
        plt.scatter(input.cpu().detach().numpy(), model(input).cpu().detach().numpy(), label='Predicted Data')
        # plt.scatter(P_collocation.cpu().detach().numpy(), dx_dP.cpu().detach().numpy(), label='dx_dP')
        # plt.scatter(P_collocation.cpu().detach().numpy(), approx.cpu().detach().numpy(), label='-alpha * x')
        plt.xlabel('RSSI P (dBm)')
        plt.ylabel('Distance x (m)')
        plt.legend()
        plt.show()

    @staticmethod
    def collocation_points(device, requires_grad=True):
        # Collocation points for the physics loss representing RSSI with normal distribution
        # P_collocation = torch.randn(1000, 1, device=device, requires_grad=requires_grad)
        P_collocation = torch.linspace(-120, -10, 1000, device=device, requires_grad=requires_grad).view(-1, 1)
        return P_collocation

    def forward(self, model, input, target, physics_data=None):
        if self.lambda_physics == 0:
            output = model(input)
            total_loss = self.data_loss_fn(output, target)
        
        else:
            output = model(input)
            data_loss = self.data_loss_fn(output, target)  # Data prediction loss

            # Collocation points for the physics loss representing RSSI with uniform distribution
            # P_collocation = torch.linspace(-3, 3, 1000, device=input.device, requires_grad=True).view(-1, 1)

            # Collocation points for the physics loss representing RSSI with normal distribution
            P_collocation = self.collocation_points(device=input.device, requires_grad=True)
            
            # Physics loss: enforce the differential equation
            # Compute the derivative dx/dP using autograd,
            # where x represents the distance and P the RSSI in dBm
            x_pred_collocation = model(P_collocation)
            dx_dP = torch.autograd.grad(
                outputs=x_pred_collocation,
                inputs=P_collocation,
                grad_outputs=torch.ones_like(x_pred_collocation),  # dLoss/dx = 1, dLoss/dP = dLoss/dx * dx/dP
                create_graph=True
            )[0]
            
            # Differential equation: dx/dP + (ln(10)/(10*alpha)) * x = 0 if data not standardized
            # x = sigma_x * x* + mu_x, P = sigma_P * P* + mu_P, x* standardized distance, P* standardized RSSI
            # dx/dP = d(sigma_x * x* + mu_x) / d(sigma_P * P* + mu_P) = (sigma_x / sigma_P) * dx*/dP*
            # alpha = physics_data[0, 0]
            # x_pred_collocation = self.std_target * x_pred_collocation + self.mean_target
            # dx_dP = (self.std_target / self.std_input) * dx_dP
            # k = torch.log(torch.tensor(10.0)) / (10 * model.alpha)
            # residual = dx_dP + k * x_pred_collocation

            residual = dx_dP + torch.einsum('f,bf->bf', model.path_loss, x_pred_collocation)

            # # print all the shapes
            # print(f"dx_dP: {dx_dP.shape}, x_pred_collocation: {x_pred_collocation.shape}, residual: {residual.shape}, alpha: {alpha.shape}")

            physics_loss = torch.mean(residual**2)

            # Initial condition: x(P0) = x0 > 0, i.e. x(-50) = 1
            # P0 = (torch.tensor(-50.0, device=input.device).view(-1,1) - self.mean_input) / self.std_input
            # x0 = (torch.tensor(1.0, device=input.device).view(-1,1) - self.mean_target) / self.std_target
            # x0_pred = model(P0)
            # physics_loss += (x0_pred - x0)**2

            # Compute total loss
            total_loss = self.lambda_data * data_loss + self.lambda_physics * physics_loss
            # print(f"Data loss: {data_loss.item()}, Physics loss: {physics_loss.item()}")
        
        return total_loss