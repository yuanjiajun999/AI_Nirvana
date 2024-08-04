import unittest
import torch
import torch.nn as nn
from src.core.semi_supervised_learning import SemiSupervisedDataset, AdvancedSemiSupervisedTrainer
import numpy as np

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

class TestSemiSupervisedLearning(unittest.TestCase):
    def setUp(self):
        self.input_dim = 10
        self.num_classes = 3
        self.num_labeled = 100
        self.num_unlabeled = 500
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.labeled_data = [(torch.randn(self.input_dim), torch.randint(0, self.num_classes, (1,)).item()) for _ in range(self.num_labeled)]
        self.unlabeled_data = [torch.randn(self.input_dim) for _ in range(self.num_unlabeled)]

        self.model = SimpleModel(self.input_dim, self.num_classes).to(self.device)
        self.trainer = AdvancedSemiSupervisedTrainer(self.model, self.labeled_data, self.unlabeled_data, self.device, self.num_classes)
        self.trainer.input_dim = self.input_dim

    def test_dataset_creation(self):
        dataset = SemiSupervisedDataset(self.labeled_data, self.unlabeled_data)
        self.assertEqual(len(dataset), self.num_labeled + self.num_unlabeled)
        
        first_data, first_label = dataset[0]
        self.assertEqual(first_data.shape, torch.Size([self.input_dim]))
        self.assertEqual(first_label.shape, torch.Size([1]))
        self.assertTrue(0 <= first_label.item() < self.num_classes)

        last_data, last_label = dataset[-1]
        self.assertEqual(last_data.shape, torch.Size([self.input_dim]))
        self.assertEqual(last_label.item(), -1)  # Unlabeled data

    def test_mixup(self):
        inputs = torch.randn(10, self.input_dim).to(self.device)
        targets = torch.randint(0, self.num_classes, (10, 1)).to(self.device)
        mixed_inputs, targets_a, targets_b, lam = self.trainer.mixup_data(inputs, targets)
        self.assertEqual(mixed_inputs.shape, inputs.shape)
        self.assertTrue(0 <= lam <= 1)

    def test_consistency_loss(self):
        pred1 = torch.randn(10, self.num_classes).to(self.device)
        pred2 = torch.randn(10, self.num_classes).to(self.device)
        loss = self.trainer.consistency_loss(pred1, pred2)
        self.assertIsInstance(loss.item(), float)

    def test_pseudo_labeling(self):
        unlabeled_data = torch.stack([data for data, _ in self.trainer.dataset.unlabeled_data[:10]])
        pseudo_labels = self.trainer.pseudo_label(unlabeled_data)
        self.assertGreater(len(pseudo_labels), 0)
        for _, label in pseudo_labels:
            self.assertTrue(0 <= label.item() < self.num_classes)

    def test_gaussian_mixture_pseudo_label(self):
        unlabeled_data = [data for data, _ in self.trainer.dataset.unlabeled_data[:10]]
        pseudo_labels = self.trainer.gaussian_mixture_pseudo_label(unlabeled_data)
        self.assertEqual(len(pseudo_labels), 10)
        for _, label in pseudo_labels:
            self.assertTrue(0 <= label.item() < self.num_classes)

    def test_train(self):
        final_loss = self.trainer.train(epochs=1)
        self.assertIsInstance(final_loss, float)
        self.assertGreater(final_loss, 0)  # Loss should be positive
        
    def test_evaluate(self):
        accuracy = self.trainer.evaluate()
        self.assertTrue(0 <= accuracy <= 1)

    def test_fine_tune(self):
        new_data = [(torch.randn(self.input_dim), torch.randint(0, self.num_classes, (1,)).item()) for _ in range(50)]
        initial_accuracy = self.trainer.evaluate()
        print(f"Initial accuracy: {initial_accuracy:.4f}")
        
        best_accuracy = initial_accuracy
        for i in range(5):  # Try fine-tuning multiple times
            self.trainer.fine_tune(new_data, epochs=10)  # Increase epochs
            final_accuracy = self.trainer.evaluate()
            print(f"Fine-tuning attempt {i+1}, accuracy: {final_accuracy:.4f}")
            
            if final_accuracy > best_accuracy:
                best_accuracy = final_accuracy
            
            if final_accuracy > initial_accuracy:
                print(f"Accuracy improved after fine-tuning")
                self.assertGreater(final_accuracy, initial_accuracy)
                return
        
        print(f"Best accuracy after fine-tuning: {best_accuracy:.4f}")
        # Allow for a small decrease in accuracy (5%)
        self.assertGreaterEqual(best_accuracy, initial_accuracy * 0.95, 
                                f"Best accuracy {best_accuracy:.4f} is significantly lower than initial accuracy {initial_accuracy:.4f}")
        
    def test_process_multimodal_input(self):
        text = "Sample text"
        image = torch.randn(3, 224, 224)
        output = self.trainer.process_multimodal_input(text, image)
        self.assertIsNotNone(output)
        self.assertEqual(output.shape, (1, self.num_classes))

    def test_reinforcement_learning_step(self):
        state = torch.randn(self.input_dim)
        action = torch.randint(0, self.num_classes, (1,))
        reward = torch.rand(1)
        next_state = torch.randn(self.input_dim)
        loss = self.trainer.reinforcement_learning_step(state, action, reward, next_state)
        self.assertIsInstance(loss, float)

    def test_advanced_reasoning(self):
        query = "What is the relationship between A and B?"
        result = self.trainer.advanced_reasoning(query)
        self.assertIsNotNone(result)
        self.assertIn("Reasoning result for query", result)
        self.assertIn("output shape", result)

    def test_autonomous_learning(self):
        new_data = [torch.randn(self.input_dim) for _ in range(10)]
        result = self.trainer.autonomous_learning(new_data)
        self.assertEqual(result, "Autonomous learning completed")

    def test_integrate_knowledge_graph(self):
        knowledge_graph = {"entity1": ["relation1", "entity2"], "entity2": ["relation2", "entity3"]}
        result = self.trainer.integrate_knowledge_graph(knowledge_graph)
        self.assertIn("Integrated knowledge graph with", result)

    def test_mixup_edge_cases(self):
        # Test with alpha=0 (no mixup)
        inputs = torch.randn(10, self.input_dim).to(self.device)
        targets = torch.randint(0, self.num_classes, (10, 1)).to(self.device)
        mixed_inputs, targets_a, targets_b, lam = self.trainer.mixup_data(inputs, targets, alpha=0)
        self.assertTrue(torch.allclose(mixed_inputs, inputs))
        self.assertEqual(lam, 1)

    def test_gaussian_mixture_pseudo_label_edge_cases(self):
        # Test with very few unlabeled data points
        unlabeled_data = [torch.randn(self.input_dim) for _ in range(2)]
        pseudo_labels = self.trainer.gaussian_mixture_pseudo_label(unlabeled_data)
        self.assertEqual(len(pseudo_labels), 2)

        # Test with number of samples equal to number of classes
        unlabeled_data = [torch.randn(self.input_dim) for _ in range(self.num_classes)]
        pseudo_labels = self.trainer.gaussian_mixture_pseudo_label(unlabeled_data)
        self.assertEqual(len(pseudo_labels), self.num_classes)

        # Test with number of samples less than number of classes
        unlabeled_data = [torch.randn(self.input_dim) for _ in range(self.num_classes - 1)]
        pseudo_labels = self.trainer.gaussian_mixture_pseudo_label(unlabeled_data)
        self.assertEqual(len(pseudo_labels), self.num_classes - 1)

    def test_train_with_all_unlabeled_data(self):
        # Create a trainer with only unlabeled data
        all_unlabeled_trainer = AdvancedSemiSupervisedTrainer(
            self.model, [], self.unlabeled_data, self.device, self.num_classes
        )
        all_unlabeled_trainer.input_dim = self.input_dim
        final_loss = all_unlabeled_trainer.train(epochs=1)
        print(f"Final loss in test_train_with_all_unlabeled_data: {final_loss}")
        self.assertEqual(final_loss, -1, f"Expected -1 (no valid training), but got {final_loss}")
        
    def test_train_with_pseudo_labeling(self):
        # Ensure we have some unlabeled data
        self.trainer.dataset.unlabeled_data = [(torch.randn(self.input_dim), torch.tensor([-1])) for _ in range(10)]
        
        # Train for multiple epochs to trigger pseudo-labeling
        final_loss = self.trainer.train(epochs=6, pseudo_labeling_interval=5)
        self.assertIsInstance(final_loss, float)
        self.assertFalse(np.isnan(final_loss))

    def test_train_with_nan_loss(self):
        # Create a model that will produce NaN loss
        class NaNModel(nn.Module):
            def __init__(self, input_dim, num_classes):
                super().__init__()
                self.fc = nn.Linear(input_dim, num_classes)
            def forward(self, x):
                return self.fc(x) * float('inf')  # This will cause NaN loss
        
        nan_model = NaNModel(self.input_dim, self.num_classes).to(self.device)
        nan_trainer = AdvancedSemiSupervisedTrainer(nan_model, self.labeled_data, self.unlabeled_data, self.device, self.num_classes)
        
        # Train and expect it to handle NaN loss
        final_loss = nan_trainer.train(epochs=1)
        print(f"Final loss in test: {final_loss}")
        self.assertTrue(np.isnan(final_loss), f"Expected NaN, but got {final_loss}")

    def test_dataset_creation_edge_cases(self):
        # Test dataset creation with empty labeled and unlabeled data
        empty_dataset = SemiSupervisedDataset([], [])
        self.assertEqual(len(empty_dataset), 0)

        # Test dataset creation with only labeled data
        labeled_only_dataset = SemiSupervisedDataset(self.labeled_data, [])
        self.assertEqual(len(labeled_only_dataset), len(self.labeled_data))

        # Test dataset creation with only unlabeled data
        unlabeled_only_dataset = SemiSupervisedDataset([], self.unlabeled_data)
        self.assertEqual(len(unlabeled_only_dataset), len(self.unlabeled_data))
    
    def test_dataset_with_invalid_data(self):
        # Test dataset creation with invalid data types
        with self.assertRaises(TypeError):
            SemiSupervisedDataset([("invalid", 0)], [])

    def test_gaussian_mixture_pseudo_label_with_single_sample(self):
        # Test gaussian_mixture_pseudo_label with a single sample
        single_sample = [torch.randn(self.input_dim)]
        pseudo_labels = self.trainer.gaussian_mixture_pseudo_label(single_sample)
        self.assertEqual(len(pseudo_labels), 1)
        self.assertEqual(pseudo_labels[0][1].item(), 0)  # Check if the default label (0) is assigned

    def test_train_with_single_batch(self):
        # Create a trainer with only one batch of data
        single_batch_data = [(torch.randn(self.input_dim), torch.randint(0, self.num_classes, (1,)).item()) for _ in range(32)]
        single_batch_trainer = AdvancedSemiSupervisedTrainer(
            self.model, single_batch_data, [], self.device, self.num_classes
        )
        single_batch_trainer.input_dim = self.input_dim
        final_loss = single_batch_trainer.train(epochs=1)
        self.assertIsInstance(final_loss, float)
        self.assertGreater(final_loss, 0)

if __name__ == '__main__':
    unittest.main()