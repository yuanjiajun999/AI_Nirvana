import torch
import torch.nn as nn
from src.core.semi_supervised_learning import SemiSupervisedDataset, AdvancedSemiSupervisedTrainer

# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

    def extract_features(self, x):
        return torch.relu(self.fc1(x))

# Set up parameters
input_dim = 10
num_classes = 3
num_labeled = 100
num_unlabeled = 500
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generate dummy data
labeled_data = [(torch.randn(input_dim), torch.randint(0, num_classes, (1,)).item()) for _ in range(num_labeled)]
unlabeled_data = [torch.randn(input_dim) for _ in range(num_unlabeled)]

# Create model and trainer
model = SimpleModel(input_dim, num_classes).to(device)
trainer = AdvancedSemiSupervisedTrainer(model, labeled_data, unlabeled_data, device, num_classes)

# Train the model
epochs = 10
final_loss = trainer.train(epochs=epochs)
print(f"Training completed. Final loss: {final_loss}")

# Evaluate the model
accuracy = trainer.evaluate()
print(f"Model accuracy: {accuracy}")

# Fine-tune the model
new_data = [(torch.randn(input_dim), torch.randint(0, num_classes, (1,)).item()) for _ in range(50)]
trainer.fine_tune(new_data, epochs=5)

# Re-evaluate after fine-tuning
new_accuracy = trainer.evaluate()
print(f"Model accuracy after fine-tuning: {new_accuracy}")

# Generate pseudo-labels for unlabeled data
pseudo_labeled_data = trainer.gaussian_mixture_pseudo_label(unlabeled_data)
print(f"Generated {len(pseudo_labeled_data)} pseudo-labels")

# Use the model for predictions
test_input = torch.randn(1, input_dim).to(device)
with torch.no_grad():
    prediction = model(test_input)
    predicted_class = torch.argmax(prediction, dim=1).item()
print(f"Predicted class for test input: {predicted_class}")