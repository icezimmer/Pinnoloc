from torch.utils.data import Dataset
import torch

def compute_mean_std(dataloader):
    """
    Compute the mean and standard deviation of the input and target data.
    - Input:
        - dataloader (DataLoader): DataLoader object containing (input_, target).
    - Output:
        - mean_input (Tensor): Mean of the input data (per feature).
        - std_input (Tensor): Standard deviation of the input data (per feature).
        - mean_target (Tensor): Mean of the target data (per feature or scalar).
        - std_target (Tensor): Standard deviation of the target data.
    """
    
    # Initialize accumulators. 
    # If you expect inputs of shape [B, D], you can init as zeros of shape [D].
    # Or, start them as 0 and let PyTorch convert to float automatically.
    running_sum_input = 0.0
    running_sum_sq_input = 0.0
    running_sum_target = 0.0
    running_sum_sq_target = 0.0
    count = 0

    for item in dataloader:
        input_, target = item[0], item[1]
        # Sum over the batch dimension to get per-feature sums
        running_sum_input += input_.sum(dim=0)
        running_sum_sq_input += (input_ ** 2).sum(dim=0)
        running_sum_target += target.sum(dim=0)
        running_sum_sq_target += (target ** 2).sum(dim=0)
        
        # Increase count by the batch size
        count += input_.size(0)

    # Mean and variance for the inputs
    mean_input = running_sum_input / count
    var_input = (running_sum_sq_input / count) - (mean_input ** 2)
    std_input = torch.sqrt(var_input + 1e-8)  # epsilon to avoid NaNs

    # Mean and variance for the targets
    mean_target = running_sum_target / count
    var_target = (running_sum_sq_target / count) - (mean_target ** 2)
    std_target = torch.sqrt(var_target + 1e-8)

    return mean_input, std_input, mean_target, std_target


class StandardizeDataset(Dataset):
    def __init__(self, base_dataset, mean_input, std_input, mean_target=None, std_target=None):
        self.base_dataset = base_dataset
        self.mean_input = mean_input
        self.std_input = std_input
        self.mean_target = mean_target
        self.std_target = std_target

    @staticmethod
    def standardize(x, mean, std):
        return (x - mean) / std
    
    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        if len(item) == 3:
            x, y, physics = item
            x_std = self.standardize(x, self.mean_input, self.std_input)
            if self.mean_target is not None and self.std_target is not None:
                y_std = self.standardize(y, self.mean_target, self.std_target)
            else:
                y_std = y

            return x_std, y_std, physics
        else:
            x, y = item
            x_std = self.standardize(x, self.mean_input, self.std_input)
            if self.mean_target is not None and self.std_target is not None:
                y_std = self.standardize(y, self.mean_target, self.std_target)
            else:
                y_std = y
            return x_std, y_std
    
    def __len__(self):
        return len(self.base_dataset)
