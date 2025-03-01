import os
import matplotlib.pyplot as plt
import json
import torch
import numpy as np
from torchmetrics import Accuracy, Precision, Recall, F1Score, ConfusionMatrix, AUROC
from tqdm import tqdm
from Pinnoloc.utils.check_device import check_model_device
from sklearn.metrics import accuracy_score, confusion_matrix

class EvaluateClassifier:
    def __init__(self, model, num_classes, dataloader, average='macro', task="multiclass"):
        self.model = model
        self.device = check_model_device(model=self.model)
        self.dataloader = dataloader

        # Add `task` argument as required by torchmetrics
        self.accuracy = Accuracy(task=task, num_classes=num_classes).to(self.device)
        self.precision = Precision(task=task, num_classes=num_classes, average=average).to(self.device)
        self.recall = Recall(task=task, num_classes=num_classes, average=average).to(self.device)
        self.f1 = F1Score(task=task, num_classes=num_classes, average=average).to(self.device)
        self.roc_auc = AUROC(task=task, num_classes=num_classes).to(self.device)  # `average` removed for AUROC
        self.confusion_matrix = ConfusionMatrix(task=task, num_classes=num_classes).to(self.device)

        # Metric values
        self.accuracy_value = None
        self.precision_value = None
        self.recall_value = None
        self.f1_value = None
        self.roc_auc_value = None
        self.confusion_matrix_value = None

    def _predict(self, verbose):
        for input_, target in tqdm(self.dataloader, disable=~verbose):
            input_ = input_.to(self.device)
            target = target.to(self.device)

            output = self.model(input_)  # Model outputs logits
            predictions = torch.argmax(output, dim=1)  # Convert logits to class predictions

            # Update metrics with correct inputs
            self.accuracy.update(predictions, target)
            self.precision.update(predictions, target)
            self.recall.update(predictions, target)
            self.f1.update(predictions, target)
            self.roc_auc.update(output, target)  # Use logits, not predictions!
            self.confusion_matrix.update(predictions, target)

    def evaluate(self, verbose=True, saving_path=None):
        self.model.eval()
        with torch.no_grad():
            self._predict()

        # Compute final metric values
        self.accuracy_value = self.accuracy.compute().item()
        self.precision_value = self.precision.compute().item()
        self.recall_value = self.recall.compute().item()
        self.f1_value = self.f1.compute().item()
        self.roc_auc_value = self.roc_auc.compute().item()
        self.confusion_matrix_value = self.confusion_matrix.compute()

        if verbose:
            print(f"Accuracy: {self.accuracy_value}")
            print(f"Precision: {self.precision_value}")
            print(f"Recall: {self.recall_value}")
            print(f"F1-Score: {self.f1_value}")
            print(f"ROC-AUC Score: {self.roc_auc_value}")
            print("Confusion Matrix:\n", self.confusion_matrix_value.cpu().numpy())

        if saving_path is not None:
            self._plot(saving_path)

    def reset(self):
        self.accuracy.reset()
        self.precision.reset()
        self.recall.reset()
        self.f1.reset()
        self.roc_auc.reset()
        self.confusion_matrix.reset()

        self.accuracy_value = None
        self.precision_value = None
        self.recall_value = None
        self.f1_value = None
        self.roc_auc_value = None
        self.confusion_matrix_value = None

    def _plot(self, saving_path):
        metrics_path = os.path.join(saving_path, 'metrics.json')
        confusion_matrix_path = os.path.join(saving_path, 'confusion_matrix.png')

        metrics = {
            "Accuracy": self.accuracy_value,
            "Precision": self.precision_value,
            "Recall": self.recall_value,
            "F1-Score": self.f1_value,
            "ROC-AUC": self.roc_auc_value
        }

        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)

        # Plot Confusion Matrix
        fig, ax = plt.subplots()
        cax = ax.matshow(self.confusion_matrix_value.cpu().numpy(), cmap=plt.cm.Purples)
        plt.title('Confusion Matrix')
        fig.colorbar(cax)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(confusion_matrix_path)
        plt.close()


class EvaluateRegressor:
    def __init__(self, model, dataloader):
        self.model = model
        self.device = check_model_device(model=self.model)
        self.dataloader = dataloader

        # Metric values
        self.mse = None
        self.rmse = None
        self.p50 = None
        self.p75 = None
        self.p90 = None
        self.mae = None
        self.min_ae = None
        self.max_ae = None


    def _predict(self, verbose):
        predictions = []
        targets = []
        
        for input_, target in tqdm(self.dataloader, disable=~verbose):            
            input_ = input_.to(self.device)
            target = target.to(self.device)

            output = self.model(input_)  # Model outputs continuous values (not logits)
            predictions.append(output)
            targets.append(target)

        # Concatenate all tensors into one for proper calculations
        return torch.cat(predictions, dim=0), torch.cat(targets, dim=0)

    def evaluate(self, mean=None, std=None, verbose=True, saving_path=None):
        self.model.eval()
        with torch.no_grad():
            predictions, targets = self._predict(verbose)

        # If mean and std are provided, unnormalize the data
        if mean is not None and std is not None:
            predictions = predictions * std + mean
            targets = targets * std + mean

        # Compute the Euclidean distances between predictions and targets
        distances = torch.norm(predictions - targets, p=2, dim=1)  # Shape: (N,)

        # Compute the 50th, 75th, and 90th percentile
        quantiles = torch.quantile(distances, torch.tensor([0.50, 0.75, 0.90]))

        self.mse    = distances.pow(2).mean().item()         # mean of squared distances
        self.rmse   = distances.pow(2).mean().sqrt().item()  # root mean squared error
        self.p50    = quantiles[0].item()                    # 50th percentile
        self.p75    = quantiles[1].item()                    # 75th percentile
        self.p90    = quantiles[2].item()                    # 90th percentile
        self.mae    = distances.mean().item()                # mean of distances (Euclidean)
        self.min_ae = distances.min().item()                 # min distance
        self.max_ae = distances.max().item()                 # max distance

        if verbose:
            print(f"Mean Squared Error (MSE): {self.mse:.6f}")
            print(f"Root Mean Squared Error (RMSE): {self.rmse:.6f}")
            print(f"50th Percentile: {self.p50:.6f}")
            print(f"75th Percentile: {self.p75:.6f}")
            print(f"90th Percentile: {self.p90:.6f}")
            print(f"Mean Absolute Error (MAE): {self.mae:.6f}")
            print(f"Min Absolute Error (MinAE): {self.min_ae:.6f}")
            print(f"Max Absolute Error (MaxAE): {self.max_ae:.6f}")

        if saving_path is not None:
            self._plot(saving_path)

    def reset(self):
        """Resets the stored metric values."""
        self.mse = None
        self.rmse = None
        self.p50 = None
        self.p75 = None
        self.p90 = None
        self.mae = None
        self.min_ae = None
        self.max_ae = None

    def _plot(self, saving_path):
        """Save metrics and plot results."""
        metrics_path = os.path.join(saving_path, 'metrics.json')
        plot_path = os.path.join(saving_path, 'regression_plot.png')

        metrics = {
            "Mean Squared Error": self.mse,
            "Root Mean Squared Error": self.rmse,
            "50th Percentile": self.p50,
            "75th Percentile": self.p75,
            "90th Percentile": self.p90,
            "Mean Absolute Error": self.mae,
            "Min Absolute Error": self.min_ae,
            "Max Absolute Error": self.max_ae
        }

        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
