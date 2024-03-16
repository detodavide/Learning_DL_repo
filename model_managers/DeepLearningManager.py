from pydantic import BaseModel
from torch.nn import Module
import torch
import torch.cuda as cuda
from torch.optim import Optimizer
import logging
import numpy as np
import random


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


class DeepLearningManager(BaseModel):
    model: Module
    loss_fn: Module
    optim: Optimizer
    epochs: int
    train_losses: list
    test_losses: list

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
            data_loader = self.train_loader
            step_fn = self.train_step_fn
        else:
            data_loader = self.val_loader
            step_fn = self.val_step_fn

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
                f"Epoch: {epoch}/{n_epochs} || Train Loss: {train_loss} || Val Loss: {val_loss}"
            )

    def predict(self, x):

        logger.debug("Performing prediction...")
        self.model.eval()
        x = torch.as_tensor(x).float()
        y_pred = self.model(x.to(self.device))
        self.model.train()
        logger.debug("Prediction completed.")
        return y_pred.detach().cpu().numpy()

    def save_checkpoints():
        pass

    def load_checlpoints():
        pass
