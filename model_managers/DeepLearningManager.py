from pydantic import BaseModel
from torch.nn import Module
import torch
import torch.cuda as cuda
import torch.nn as nn
from torch.optim import Optimizer
import logging
import numpy as np
import random
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


class DeepLearningManager:

    def __init__(
        self,
        model: Module,
        loss_function: Module,
        optimizer: Optimizer,
    ) -> None:
        super().__init__()
        self.model = model
        self.loss_fn = loss_function
        self.optim = optimizer

        self.device = "cuda" if cuda.is_available() else "cpu"
        self.model.to(self.device)

        # Data Loader
        self.train_loader = None
        self.val_loader = None

        # Evaluation
        self.train_losses: list = []
        self.test_losses: list = []
        self.epochs: int = 0

        # Training and Validation step functions
        self.train_step_fn = self._make_train_step_fn()
        self.val_step_fn = self._make_val_step_fn()

    def to(self, device):
        try:
            self.device = device
            self.model.to(self.device)
        except Exception as e:
            logger.info(e)

    def set_data_loaders(self, train_loader, val_loader=None):
        self.train_loader = train_loader
        self.val_loader = val_loader

    def set_seed(self, seed=101):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        try:
            self.train_loader.sampler.generator.manual_seed(seed)
        except AttributeError:
            pass

    def _make_train_step_fn(self):
        def perform_train_step_fn(x, y):
            self.model.train()
            y_pred = self.model(x)
            loss = self.loss_fn(y_pred, y)
            loss.backward()

            self.optim.step()
            self.optim.zero_grad()
            return loss.item()

        return perform_train_step_fn

    def _make_val_step_fn(self):
        def perform_val_step_fn(x, y):
            self.model.eval()
            y_pred = self.model(x)
            loss = self.loss_fn(y_pred, y)

            return loss.item()

        return perform_val_step_fn

    def _mini_batch(self, validation=False):
        if validation:
            data_loader = self.val_loader
            step_fn = self.val_step_fn
        else:
            data_loader = self.train_loader
            step_fn = self.train_step_fn

        if data_loader is None:
            return None

        mini_batch_losses = []
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            mini_batch_loss = step_fn(x_batch, y_batch)
            mini_batch_losses.append(mini_batch_loss)

        loss = np.mean(mini_batch_losses)

        return loss

    def train(self, n_epochs, seed=101):

        self.set_seed(seed)
        for epoch in range(n_epochs):
            self.epochs += 1
            train_loss = self._mini_batch(validation=False)
            self.train_losses.append(train_loss)

            with torch.no_grad():
                val_loss = self._mini_batch(validation=True)
                self.test_losses.append(val_loss)

            logger.info(
                f"Epoch: {epoch+1}/{n_epochs} || Train Loss: {train_loss} || Val Loss: {val_loss}"
            )

    def save_checkpoint(self, filename):
        checkpoint = {
            "epoch": self.total_epochs,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": self.losses,
            "val_loss": self.val_losses,
        }

        torch.save(checkpoint, filename)

    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.total_epochs = checkpoint["epoch"]
        self.losses = checkpoint["loss"]
        self.val_losses = checkpoint["val_loss"]

        self.model.train()

    def predict(self, x):
        self.model.eval()
        x_tensor = torch.as_tensor(x).float()
        y_hat_tensor = self.model(x_tensor.to(self.device))
        self.model.train()
        return y_hat_tensor.detach().cpu().numpy()

    def plot_losses(self):
        fig = plt.figure(figsize=(10, 4))
        plt.plot(self.train_losses, label="Training Loss", c="b")
        plt.plot(self.test_losses, label="Validation Loss", c="r")
        plt.yscale("log")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        return fig

    def correct(self, x, y, threshold=0.5):
        self.model.eval()
        yhat = self.model(x.to(self.device))
        y = y.to(self.device)
        self.model.train()

        n_samples, n_dims = yhat.shape
        if n_dims > 1:
            _, predicted = torch.max(yhat, 1)
        else:
            n_dims += 1
            if isinstance(self.model, nn.Sequential) and isinstance(
                self.model[-1], nn.Sigmoid
            ):
                predicted = (yhat > threshold).long()
            else:
                predicted = (torch.sigmoid(yhat) > threshold).long()

        result = []
        for c in range(n_dims):
            n_class = (y == c).sum().item()
            n_correct = (predicted[y == c] == c).sum().item()
            result.append((n_correct, n_class))
        return torch.tensor(result)

    @staticmethod
    def loader_apply(loader, func, reduce="sum"):
        results = [func(x, y) for i, (x, y) in enumerate(loader)]
        results = torch.stack(results, axis=0)

        if reduce == "sum":
            results = results.sum(axis=0)
        elif reduce == "mean":
            results = results.float().mean(axis=0)

        return results
