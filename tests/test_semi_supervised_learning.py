import unittest
import torch
from src.core.semi_supervised_learning import SemiSupervisedDataset, SemiSupervisedTrainer

class TestSemiSupervisedLearning(unittest.TestCase):
    def test_semi_supervised_dataset(self):
        labeled_data = [(torch.randn(10), torch.randint(0, 2, (1,))) for _ in range(50)]
        unlabeled_data = [torch.randn(10) for _ in range(100)]
        dataset = SemiSupervisedDataset(labeled_data, unlabeled_data)
        self.assertEqual(len(dataset), 150)

    def test_semi_supervised_trainer(self):
        model = torch.nn.Linear(10, 2)
        labeled_data = [(torch.randn(10), torch.randint(0, 2, (1,))) for _ in range(50)]
        unlabeled_data = [torch.randn(10) for _ in range(100)]
        trainer = SemiSupervisedTrainer(model, labeled_data, unlabeled_data, 'cpu')
        trainer.train(epochs=1)
        self.assertTrue(True)  # If we reach here without errors, the test passes

if __name__ == '__main__':
    unittest.main()