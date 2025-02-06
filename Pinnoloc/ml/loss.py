import torch


class PhysicsInformedLoss(nn.Module):
    def __init__(self, lambda_rss=0.1, lambda_aoa=0.1, A_max=10):
        super(PhysicsInformedLoss, self).__init__()
        self.lambda_rss = lambda_rss
        self.lambda_aoa = lambda_aoa
        self.A_max = A_max
        self.mse_loss = torch.nn.MSELoss()

    def forward(self, pred_theta, true_theta, rssi, aoa, distance, heading, anchor_angles):
        mse_loss = self.mse_loss(pred_theta, true_theta)  # Heading prediction loss

        # Blocking detection (1 if blocked, 0 otherwise)
        block_condition = (torch.abs(heading - anchor_angles) > torch.pi / 2).float()

        # RSSI loss
        path_loss_pred = -10 * torch.log10(distance) - block_condition * self.A_max
        rss_loss = torch.mean((rssi - path_loss_pred) ** 2)

        # AoA loss
        d_ant = 0.05  # Antenna spacing (50mm)
        lambda_signal = 0.125  # BLE 2.4 GHz wavelength
        theta_body_error = torch.randn_like(pred_theta) * block_condition * 5  # Add AoA error if blocked
        phi_pred = (2 * torch.pi * d_ant / lambda_signal) * torch.sin(pred_theta + theta_body_error)
        aoa_loss = torch.mean((aoa - phi_pred) ** 2)

        return mse_loss + self.lambda_rss * rss_loss + self.lambda_aoa * aoa_loss
