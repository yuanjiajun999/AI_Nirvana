import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class SemiSupervisedDataset(Dataset):
    def __init__(self, labeled_data, unlabeled_data):
        self.labeled_data = labeled_data
        self.unlabeled_data = unlabeled_data

    def __len__(self):
        return len(self.labeled_data) + len(self.unlabeled_data)

    def __getitem__(self, index):
        if index < len(self.labeled_data):
            return self.labeled_data[index]
        else:
            return self.unlabeled_data[index - len(self.labeled_data)]

class SemiSupervisedTrainer:
    def __init__(self, model, labeled_data, unlabeled_data, device):
        self.model = model
        self.device = device
        self.dataset = SemiSupervisedDataset(labeled_data, unlabeled_data)
        self.dataloader = DataLoader(self.dataset, batch_size=32, shuffle=True)

    def train(self, epochs):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, labels in self.dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(self.dataloader)}')