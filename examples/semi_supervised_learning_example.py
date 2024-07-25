import torch
import torch.nn as nn
from src.core.semi_supervised_learning import SemiSupervisedTrainer
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split


class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(2, 10), nn.ReLU(), nn.Linear(10, 2))

    def forward(self, x):
        return self.fc(x)


def main():
    # 生成月牙形数据集
    X, y = make_moons(n_samples=1000, noise=0.3, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 将部分训练数据设为未标记
    n_labeled = 100
    X_labeled = X_train[:n_labeled]
    y_labeled = y_train[:n_labeled]
    X_unlabeled = X_train[n_labeled:]

    # 转换为 PyTorch 张量
    X_labeled = torch.FloatTensor(X_labeled)
    y_labeled = torch.LongTensor(y_labeled)
    X_unlabeled = torch.FloatTensor(X_unlabeled)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.LongTensor(y_test)

    model = SimpleNN()
    trainer = SemiSupervisedTrainer(
        model, list(zip(X_labeled, y_labeled)), X_unlabeled, "cpu"
    )

    print("Starting semi-supervised training...")
    trainer.train(epochs=50)

    # 评估模型
    model.eval()
    with torch.no_grad():
        test_output = model(X_test)
        predicted = torch.max(test_output, 1)[1]
        accuracy = (predicted == y_test).float().mean()
        print(f"Test Accuracy: {accuracy.item():.4f}")


if __name__ == "__main__":
    main()
