

import numpy as np
import torch
from torch.utils.data import DataLoader

from mvtec_ad import MVTecAD
from .utils import create_input_transform, create_target_transform

class Trainer:

    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        train_loader,
        val_loader,
        epochs,
        device,
        save_path,
        log_interval=10,
        save_interval=10,
        verbose=True,
        step=50,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.device = device
        self.save_path = save_path
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.verbose = verbose
        self.step = step

        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = np.inf
        self.best_val_epoch = 0
        

    def train(self):
        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            val_loss = self.val_epoch(epoch)
            self.val_losses.append(val_loss)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_epoch = epoch
                self.save_model()

            if self.verbose:
                print(
                    f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                )
    
    def train_epoch(self, epoch):
        self.model.train()
        losses = []
        for batch_idx, (x, mask, y) in enumerate(self.train_loader):
            x, mask, y = x.to(self.device), mask.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            x_hat = self.model(x, self.step)
            loss = self.loss_fn(x_hat, y)
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
            if self.verbose and batch_idx % self.log_interval == 0:
                print(
                    f"Epoch {epoch}, Train Step {self.step}, Loss: {loss.item():.4f}"
                )
        return np.mean(losses)
    
    def val_epoch(self, epoch):
        self.model.eval()
        losses = []
        with torch.no_grad():
            for batch_idx, (x, mask, y) in enumerate(self.val_loader):
                x, mask, y = x.to(self.device), mask.to(self.device), y.to(self.device)
                x_hat = self.model(x, self.step)
                loss = self.loss_fn(x_hat, y)
                losses.append(loss.item())
        return np.mean(losses)
    
    def save_model(self):
        torch.save(self.model.state_dict(), self.save_path)
        if self.verbose:
            print(f"Model saved to {self.save_path}")


def main():
    random_state = 42
    torch.manual_seed(random_state)

    transform = create_input_transform()
    target_transform = create_target_transform()

    # load data
    train_dataset = MVTecAD(
        "data",
        subset_name="bottle",
        train=True,
        transform=transform,
        mask_transform=transform,
        target_transform=target_transform,
        download=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )



if __name__ == "__main__":
    main()
