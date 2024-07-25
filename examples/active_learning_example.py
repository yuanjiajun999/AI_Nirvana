import numpy as np
from sklearn.datasets import make_classification
from src.core.active_learning import ActiveLearner


def main():
    # 创建一个模拟的分类数据集
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=2, n_redundant=10, random_state=42
    )

    # 分割数据集
    X_pool, y_pool = X[:800], y[:800]
    X_test, y_test = X[800:], y[800:]

    # 初始化主动学习器
    active_learner = ActiveLearner(X_pool, y_pool, X_test, y_test)

    # 运行主动学习循环
    initial_samples = 50
    n_iterations = 10
    samples_per_iteration = 10

    final_accuracy = active_learner.active_learning_loop(
        initial_samples, n_iterations, samples_per_iteration
    )

    print(f"Final model accuracy: {final_accuracy:.4f}")


if __name__ == "__main__":
    main()
