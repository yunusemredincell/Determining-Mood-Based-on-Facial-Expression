from dataset import FacialExpressionDataset, get_data_transforms

from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import numpy as np

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)


class FacialExpressionClassifier:
    def __init__(self, data_dir, model: nn.Module, batch_size=64, learning_rate=0.001):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.train_losses = []
        self.val_losses = []

        self.train_accuracies = []
        self.val_accuracies = []

        self.class_names = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

    def create_dataloaders(self, val_split=0.2):
        transform = get_data_transforms(image_size=64)

        full_train_dataset = FacialExpressionDataset(root_dir=f"{self.data_dir}/train")

        train_size = int((1 - val_split) * len(full_train_dataset))
        val_size = len(full_train_dataset) - train_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            full_train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )
        test_dataset = FacialExpressionDataset(root_dir=f"{self.data_dir}/test")
        train_dataset.dataset.transform = transform
        val_dataset.dataset.transform = transform
        test_dataset.transform = transform

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

        self.class_names = full_train_dataset.classes
        return train_loader, val_loader, test_loader

    def fit(self, epochs=100, unfreeze_epoch=15, transfer_learning=False, patience=15):
        train_loader, val_loader, test_loader = self.create_dataloaders()

        print(
            f"Model created with {sum(p.numel() for p in self.model.parameters())} parameters"
        )

        criterion = nn.CrossEntropyLoss()

        if transfer_learning:
            optimizer = self.model.optimizer(self.learning_rate)
        else:
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=1e-4,
            )

        # scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=8)

        best_val_acc = 0.0
        patience_counter = 0
        freeze_backbone = True

        for epoch in range(epochs):

            if transfer_learning and epoch == unfreeze_epoch and freeze_backbone:
                freeze_backbone = False
                self.model.unfreeze()
                optimizer = optim.SGD(
                    self.model.parameters(),
                    lr=self.learning_rate,
                    momentum=0.9,
                    weight_decay=1e-4,
                )

            train_loss, train_accuracy = self.train(
                train_loader, criterion, optimizer, epoch, epochs
            )
            val_loss, val_accuracy = self.validate(val_loader, criterion, epoch, epochs)
            tqdm.write(
                f"{epoch+1}/{epochs} [ Train  ]: Loss: {train_loss:.3f}, Acc: {train_accuracy:.2f}%"
            )
            tqdm.write(
                f"{epoch+1}/{epochs} [Validate]: Loss: {val_loss:.3f}, Acc: {val_accuracy:.2f}%"
            )

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_accuracy)
            self.val_accuracies.append(val_accuracy)

            # scheduler.step(val_accuracy)

            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                torch.save(self.model.state_dict(), "results/model/model.pth")
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

        self.model.load_state_dict(
            torch.load("results/model/model.pth", map_location=self.device)
        )
        self.model = self.model.to(self.device)
        return train_loader, test_loader, test_loader

    def train(
        self,
        train_loader: DataLoader,
        criterion: nn.CrossEntropyLoss,
        optimizer: torch.optim.Adam,
        epoch,
        epochs,
    ):
        running_loss = 0.0
        correct = 0
        total = 0
        self.model.train()
        progress_bar = tqdm(train_loader, desc=f"{epoch+1}/{epochs} [ Train  ]")

        for _, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * correct / total

        return train_loss, train_acc

    @torch.no_grad
    def validate(self, validate_loader, criterion, epoch, epochs):
        val_loss = 0
        correct = 0
        total = 0
        progress_bar = tqdm(validate_loader, desc=f"{epoch+1}/{epochs} [Validate]")
        self.model.eval()

        for _, (data, target) in enumerate(progress_bar):
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            val_loss += criterion(output, target).item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

        val_loss /= len(validate_loader)
        val_acc = 100.0 * correct / total

        return val_loss, val_acc
