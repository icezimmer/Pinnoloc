import json
import os
import torch


def save_data(data, file_path):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Save the data using torch.save
    torch.save(data, file_path)


def load_data(file_path, device='cpu', use_dill=False):
    """
    Load a PyTorch data file safely and move it to the specified device.

    Args:
        file_path (str): Path to the file.
        device (str): Device to map the loaded tensor ('cpu' or 'cuda').
        use_dill (bool): If True, use dill for better pickle compatibility.

    Returns:
        loaded_data: The deserialized PyTorch object.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If the file is corrupted or cannot be loaded.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: The file '{file_path}' does not exist.")

    try:
        pickle_module = dill if use_dill else torch.serialization.pickle
        loaded_data = torch.load(file_path, map_location=device, pickle_module=pickle_module)
        return loaded_data
    except Exception as e:
        raise RuntimeError(f"Error loading file '{file_path}': {e}")
