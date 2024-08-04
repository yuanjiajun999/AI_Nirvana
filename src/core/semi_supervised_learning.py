import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.mixture import GaussianMixture

class SemiSupervisedDataset(Dataset):
    def __init__(self, labeled_data, unlabeled_data, transform=None):
        self.labeled_data = [(torch.tensor(data, dtype=torch.float32), torch.tensor([label], dtype=torch.long)) for data, label in labeled_data]
        self.unlabeled_data = [(torch.tensor(data, dtype=torch.float32), torch.tensor([-1], dtype=torch.long)) for data in unlabeled_data]
        self.all_data = self.labeled_data + self.unlabeled_data
        self.transform = transform

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, index):
        data, label = self.all_data[index]
        if self.transform:
            data = self.transform(data)
        return data, label

class AdvancedSemiSupervisedTrainer:
    def __init__(self, model, labeled_data, unlabeled_data, device, num_classes):
        self.model = model
        self.device = device
        self.num_classes = num_classes
        self.dataset = SemiSupervisedDataset(labeled_data, unlabeled_data)
        self.dataloader = DataLoader(self.dataset, batch_size=32, shuffle=True, collate_fn=self.custom_collate)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)

    def custom_collate(self, batch):
        data = torch.stack([item[0] for item in batch])
        labels = torch.cat([item[1] for item in batch])
        return data, labels

    def mixup_data(self, x, y, alpha=1.0):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(self.device)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def mixup_criterion(self, pred, y_a, y_b, lam):
        return lam * self.criterion(pred, y_a) + (1 - lam) * self.criterion(pred, y_b)

    def consistency_loss(self, pred1, pred2):
        return F.mse_loss(F.softmax(pred1, dim=1), F.softmax(pred2, dim=1))

    def pseudo_label(self, unlabeled_data, threshold=0.8):
        self.model.eval()
        pseudo_labels = []
        with torch.no_grad():
            for data in unlabeled_data:
                outputs = self.model(data.unsqueeze(0).to(self.device))
                prob, pred = torch.max(F.softmax(outputs, dim=1), dim=1)
                if prob.item() > threshold:
                    pseudo_labels.append((data, torch.tensor([pred.item()], dtype=torch.long)))
        return pseudo_labels if pseudo_labels else [(unlabeled_data[0], torch.tensor([0], dtype=torch.long))]

    def gaussian_mixture_pseudo_label(self, unlabeled_data):
        self.model.eval()
        features = []
        with torch.no_grad():
            for data in unlabeled_data:
                feat = self.model.extract_features(data.unsqueeze(0).to(self.device))
                features.append(feat.cpu().numpy())
        
        features = np.concatenate(features, axis=0)
        
        if len(features) < 2:
            # If we have less than 2 samples, we can't use GMM
            # Instead, we'll just assign a default label (e.g., 0)
            return [(data, torch.tensor([0], dtype=torch.long)) for data in unlabeled_data]
        
        n_components = min(self.num_classes, len(features))
        gmm = GaussianMixture(n_components=n_components, random_state=0)
        pseudo_labels = gmm.fit_predict(features)
        
        return [(data, torch.tensor([label], dtype=torch.long)) for data, label in zip(unlabeled_data, pseudo_labels)]
    
    def train(self, epochs, pseudo_labeling_interval=5):
        final_loss = -1  # Initialize with -1 to indicate no valid training
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0
            num_batches = 0
            all_batches_nan = True
            any_valid_batch = False
            for batch_idx, (inputs, labels) in enumerate(self.dataloader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Skip batches with all -1 labels (all unlabeled data)
                if torch.all(labels == -1):
                    print(f"Skipping batch {batch_idx+1} as all labels are -1")
                    continue
                
                any_valid_batch = True
                
                # Mixup
                inputs, targets_a, targets_b, lam = self.mixup_data(inputs, labels)
                
                # Consistency regularization
                inputs_aug = inputs + 0.1 * torch.randn_like(inputs)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                outputs_aug = self.model(inputs_aug)
                
                loss = self.mixup_criterion(outputs, targets_a, targets_b, lam)
                loss += 0.1 * self.consistency_loss(outputs, outputs_aug)
                
                # Check for NaN loss
                if torch.isnan(loss):
                    print(f"NaN loss detected at epoch {epoch+1}, batch {batch_idx+1}. Skipping this batch.")
                    continue
                
                all_batches_nan = False
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                running_loss += loss.item()
                num_batches += 1
            
            self.scheduler.step()
            
            if num_batches > 0:
                avg_loss = running_loss / num_batches
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
                final_loss = avg_loss
            else:
                print(f"Epoch [{epoch+1}/{epochs}], No valid batches")
            
            # Pseudo-labeling
            if (epoch + 1) % pseudo_labeling_interval == 0:
                pseudo_labeled_data = self.gaussian_mixture_pseudo_label([data for data, _ in self.dataset.unlabeled_data])
                self.dataset.all_data = self.dataset.labeled_data + pseudo_labeled_data
                self.dataloader = DataLoader(self.dataset, batch_size=32, shuffle=True, collate_fn=self.custom_collate)

        if not any_valid_batch:
            final_loss = -1
        elif all_batches_nan:
            final_loss = float('nan')
        print(f"Training completed. Final loss: {final_loss}")
        return final_loss

    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in self.dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += ((predicted == labels.squeeze()) & (labels != -1)).sum().item()
        accuracy = correct / total
        print(f"Test Accuracy: {accuracy:.4f}")
        return accuracy

    def fine_tune(self, new_data, epochs):
        new_dataset = SemiSupervisedDataset(new_data, [])
        new_dataloader = DataLoader(new_dataset, batch_size=32, shuffle=True, collate_fn=self.custom_collate)
        
        for epoch in range(epochs):
            self.model.train()
            for inputs, labels in new_dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels.squeeze())
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()
        
        print(f"Fine-tuning completed for {epochs} epochs")
        
    def process_multimodal_input(self, text, image):
        # 简单的多模态处理模拟
        text_embedding = torch.randn(self.input_dim // 2)  # 假设文本嵌入
        image_features = torch.randn(self.input_dim // 2)  # 假设图像特征
        combined_features = torch.cat([text_embedding, image_features])
        return self.model(combined_features.unsqueeze(0))

    def reinforcement_learning_step(self, state, action, reward, next_state):
        # 简单的强化学习步骤模拟
        state_value = self.model(state.unsqueeze(0))
        next_state_value = self.model(next_state.unsqueeze(0))
        td_error = reward + 0.99 * next_state_value.sum() - state_value.sum()
        loss = td_error.pow(2)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def autonomous_learning(self, new_data):
        # 简单的自主学习模拟
        self.fine_tune([(data, self.pseudo_label([data])[0][1]) for data in new_data], epochs=1)
        return "Autonomous learning completed"

    def integrate_knowledge_graph(self, knowledge_graph):
        # 简单的知识图谱集成模拟
        num_entities = len(knowledge_graph)
        knowledge_embedding = torch.randn(num_entities, 64)  # 假设每个实体的嵌入
        return f"Integrated knowledge graph with {num_entities} entities"

    def advanced_reasoning(self, query):
        # 简单的高级推理模拟
        query_embedding = torch.randn(self.input_dim)  # 假设查询的嵌入
        reasoning_output = self.model(query_embedding.unsqueeze(0))
        return f"Reasoning result for query: {query}, output shape: {reasoning_output.shape}"
