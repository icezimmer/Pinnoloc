import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from Pinnoloc.utils.check_device import check_model_device
import copy


class TrainModel:
    def __init__(self, model, optimizer, criterion, develop_dataloader):
        self.model = model
        self.device = check_model_device(model=self.model)
        self.optimizer = optimizer
        self.criterion = criterion
        self.develop_dataloader = develop_dataloader
        self.training_loss = []
        self.validation_loss = []

    def _epoch(self, dataloader, verbose):
        self.model.train()

        running_loss = 0.0
        for input_, target in tqdm(dataloader, disable=~verbose):
            self.optimizer.zero_grad()
            input_ = input_.to(self.device)
            target = target.to(self.device)
            output = self.model(input_)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(dataloader)

    def _validate(self, dataloader, verbose):
        self.model.eval()
        with torch.no_grad():
            running_loss = 0.0
            for input_, target in tqdm(dataloader, disable=~verbose):
                input_ = input_.to(self.device)
                target = target.to(self.device)
                output = self.model(input_)
                running_loss += self.criterion(output, target).item()

            return running_loss / len(dataloader)

    def max_epochs(self, num_epochs, plot_path=None):
        for epoch in range(num_epochs):
            loss_epoch = self._epoch(self.develop_dataloader)
            self.training_loss.append(loss_epoch)
            print('[%d] loss: %.3f' % (epoch + 1, loss_epoch))

        print('Finished Training')

        if plot_path is not None:
            self._plot(plot_path)

        self.training_loss = []
        self.validation_loss = []

    def early_stopping(self, train_dataloader, val_dataloader, patience, reduce_plateau, num_epochs=float('inf'),
                       verbose=False, plot_path=None):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=self.optimizer, patience=patience//2, factor=reduce_plateau)
        epoch = 0
        buffer = 0
        best_val_loss = float('inf')
        best_model_dict = self.model.state_dict()
        while buffer < patience and epoch < num_epochs:
            train_loss_epoch = self._epoch(train_dataloader, verbose)
            val_loss_epoch = self._validate(val_dataloader, verbose)
            scheduler.step(val_loss_epoch)
            self.training_loss.append(train_loss_epoch)
            self.validation_loss.append(val_loss_epoch)
            if val_loss_epoch < best_val_loss:
                buffer = 0
                best_val_loss = val_loss_epoch
                best_model_dict = copy.deepcopy(self.model.state_dict())
            else:
                buffer += 1
            print('[%d] train_loss: %.3f; val_loss: %.3f' % (epoch + 1, train_loss_epoch, val_loss_epoch))
            epoch += 1

        self.model.load_state_dict(best_model_dict)
        develop_loss = self._epoch(self.develop_dataloader)

        print('[END] develop_loss: %.3f' % develop_loss)
        print('Finished Training')

        if plot_path is not None:
            self._plot(plot_path)

    def _plot(self, plot_path):
        plt.figure(figsize=(10, 5))
        plt.plot(self.training_loss, label='Training loss')
        plt.plot(self.validation_loss, label='Validation loss')
        plt.title('Training and Validation Loss Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        plt.savefig(plot_path)  # Save the plot as a PNG file
        plt.close()  # Close the plot to free up memory


class TrainPhysicsModel:
    def __init__(self, model, optimizer, criterion, resampling_period, develop_dataloader):
        self.model = model
        self.device = check_model_device(model=self.model)
        self.optimizer = optimizer
        self.criterion = criterion
        self.resampling_period = resampling_period
        self.develop_dataloader = develop_dataloader
        self.training_loss = []
        self.validation_loss = []
    
    def reset(self):
        self.training_loss = []
        self.validation_loss = []

    def _epoch(self, dataloader, verbose):
        self.model.train()

        running_loss = 0.0
        for input_, target in tqdm(dataloader, disable=~verbose):
            self.optimizer.zero_grad()
            input_ = input_.to(self.device)
            target = target.to(self.device)
            loss = self.criterion(self.model, input_, target)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(dataloader)

    def _validate(self, dataloader, verbose):
        self.model.eval()
        with torch.no_grad():
            running_loss = 0.0
            for input_, target in tqdm(dataloader, disable=~verbose):
                input_ = input_.to(self.device)
                target = target.to(self.device)
                output = self.model(input_)
                running_loss += self.criterion.data_loss_fn(output, target).item()

            return running_loss / len(dataloader)

    def max_epochs(self, num_epochs, verbose=True, plot_path=None):
        for epoch in range(num_epochs):
            if epoch % self.resampling_period == 0:
                self.criterion.resample_collocation_points()
            loss_epoch = self._epoch(self.develop_dataloader)
            self.training_loss.append(loss_epoch)
            if verbose:
                print('[%d] loss: %.3f' % (epoch + 1, loss_epoch))

        print('Finished Training')

        if plot_path is not None:
            self._plot(plot_path)

    def early_stopping(self, train_dataloader, val_dataloader, patience, reduce_plateau, num_epochs=float('inf'),
                       verbose=True, plot_path=None):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=self.optimizer, patience=patience//2, factor=reduce_plateau)
        epoch = 0
        buffer = 0
        best_val_loss = float('inf')
        best_model_dict = self.model.state_dict()
        while buffer < patience and epoch < num_epochs:
            if epoch % self.resampling_period == 0:
                self.criterion.resample_collocation_points()
            train_loss_epoch = self._epoch(train_dataloader, verbose)
            val_loss_epoch = self._validate(val_dataloader, verbose)
            scheduler.step(val_loss_epoch)
            self.training_loss.append(train_loss_epoch)
            self.validation_loss.append(val_loss_epoch)
            if val_loss_epoch < best_val_loss:
                buffer = 0
                best_val_loss = val_loss_epoch
                best_model_dict = copy.deepcopy(self.model.state_dict())
            else:
                buffer += 1
            if verbose:
                print('[%d] train_loss: %.3f; val_loss: %.3f' % (epoch + 1, train_loss_epoch, val_loss_epoch))
            epoch += 1

        self.model.load_state_dict(best_model_dict)
        develop_loss = self._epoch(self.develop_dataloader, verbose)

        if verbose:
            print('[END] develop_loss: %.3f' % develop_loss)
            print('Finished Training')

        if plot_path is not None:
            self._plot(plot_path)

    def _plot(self, plot_path):
        plt.figure(figsize=(10, 5))
        plt.plot(self.training_loss, label='Training loss')
        plt.plot(self.validation_loss, label='Validation loss')
        plt.title('Training and Validation Loss Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        plt.savefig(plot_path)  # Save the plot as a PNG file
        plt.close()  # Close the plot to free up memory