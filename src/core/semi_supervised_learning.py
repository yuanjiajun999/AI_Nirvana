import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

class SemiSupervisedDataset(Dataset):
    """
    A dataset for semi-supervised learning that combines labeled and unlabeled data.

    Args:
        labeled_data (list): A list of tuples (data, label) for labeled samples.
        unlabeled_data (list): A list of unlabeled data samples.

    Attributes:
        all_data (list): Combined list of labeled and unlabeled data.
    """

    def __init__(self, labeled_data, unlabeled_data):
        self.labeled_data = labeled_data
        self.unlabeled_data = [(data, torch.tensor([-1])) for data in unlabeled_data]  # Use -1 to mark unlabeled data
        self.all_data = self.labeled_data + self.unlabeled_data

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.all_data)

    def __getitem__(self, index):
        """
        Retrieves a sample from the dataset.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing (data, label).
        """
        return self.all_data[index]

class SemiSupervisedTrainer:
    """
    A trainer for semi-supervised learning models.

    Args:
        model (nn.Module): The model to be trained.
        labeled_data (list): A list of tuples (data, label) for labeled samples.
        unlabeled_data (list): A list of unlabeled data samples.
        device (str): The device to use for training ('cpu' or 'cuda').

    Attributes:
        model (nn.Module): The model being trained.
        device (str): The device used for training.
        dataset (SemiSupervisedDataset): The dataset used for training.
        dataloader (DataLoader): DataLoader for batching and shuffling the dataset.
    """

    def __init__(self, model, labeled_data, unlabeled_data, device):
        self.model = model
        self.device = device
        self.dataset = SemiSupervisedDataset(labeled_data, unlabeled_data)
        self.dataloader = DataLoader(self.dataset, batch_size=32, shuffle=True)

    def train(self, epochs):
        """
        Trains the model using semi-supervised learning.

        Args:
            epochs (int): Number of epochs to train for.
        """
        criterion = nn.CrossEntropyLoss(ignore_index=-1)  # Ignore -1 labels (unlabeled data)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, labels in self.dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels.squeeze())
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(self.dataloader)}')