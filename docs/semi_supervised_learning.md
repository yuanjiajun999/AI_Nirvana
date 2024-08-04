# Semi-Supervised Learning Module Documentation

## Overview

This module provides tools for semi-supervised learning, allowing you to train models using both labeled and unlabeled data. It includes a custom dataset class and a trainer class with advanced features such as mixup data augmentation, consistency regularization, and pseudo-labeling.

## Main Components

### SemiSupervisedDataset

A custom PyTorch Dataset for handling both labeled and unlabeled data.

#### Methods:
- `__init__(self, labeled_data, unlabeled_data, transform=None)`: Initialize the dataset.
- `__len__(self)`: Return the total number of samples.
- `__getitem__(self, index)`: Retrieve a sample by index.

### AdvancedSemiSupervisedTrainer

A trainer class that implements advanced semi-supervised learning techniques.

#### Methods:
- `__init__(self, model, labeled_data, unlabeled_data, device, num_classes)`: Initialize the trainer.
- `train(self, epochs, pseudo_labeling_interval=5)`: Train the model.
- `evaluate(self)`: Evaluate the model's performance.
- `fine_tune(self, new_data, epochs)`: Fine-tune the model on new data.
- `gaussian_mixture_pseudo_label(self, unlabeled_data)`: Generate pseudo-labels for unlabeled data.
- `mixup_data(self, x, y, alpha=1.0)`: Perform mixup data augmentation.
- `consistency_loss(self, pred1, pred2)`: Calculate consistency loss for regularization.

## Usage

1. Prepare your data:
   - Labeled data should be a list of tuples: (input_tensor, label)
   - Unlabeled data should be a list of input tensors

2. Create your model:
   - The model should have a `forward` method and an `extract_features` method.

3. Initialize the trainer:
   trainer = AdvancedSemiSupervisedTrainer(model, labeled_data, unlabeled_data, device, num_classes)

4. Train the model:
   final_loss = trainer.train(epochs=10)

5. Evaluate the model:
   accuracy = trainer.evaluate()

6. Fine-tune the model (if needed):
   trainer.fine_tune(new_data, epochs=5)

7. Generate pseudo-labels for unlabeled data:
   pseudo_labeled_data = trainer.gaussian_mixture_pseudo_label(unlabeled_data)

## Best Practices

1. Data Preparation: Ensure your labeled and unlabeled data are properly preprocessed and normalized.

2. Model Selection: Choose a model architecture suitable for your specific task and data.

3. Hyperparameter Tuning: Experiment with different values for learning rate, batch size, and the number of epochs to find the optimal configuration.

4. Monitoring: Keep track of both training loss and validation accuracy to detect overfitting.

5. Pseudo-labeling: Use pseudo-labeling cautiously, as it can reinforce model biases if not used correctly.

6. Fine-tuning: When fine-tuning, start with a lower learning rate to avoid disrupting the learned features.

## Notes

- This module assumes that your model and data can fit into memory. For very large datasets, you may need to implement data loading in batches.
- The effectiveness of semi-supervised learning can vary depending on the quality and quantity of both labeled and unlabeled data.
- Always validate the model's performance on a held-out test set to ensure generalization.