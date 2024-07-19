# examples/active_learning_example.py

from src.core.active_learning import ActiveLearner
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

def main():
    # 创建一个示例数据集
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=2, n_redundant=10, random_state=42)
    
    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 初始化主动学习器
    active_learner = ActiveLearner(X_train, y_train, X_test, y_test, base_estimator=SVC(probability=True))
    
    # 运行主动学习循环
    n_iterations = 5
    samples_per_iteration = 10
    
    print("Starting active learning process...")
    for i in range(n_iterations):
        accuracy = active_learner.active_learning_step(samples_per_iteration)
        print(f"Iteration {i+1}: Accuracy = {accuracy:.4f}")
    
    final_accuracy = active_learner.evaluate()
    print(f"Final model accuracy: {final_accuracy:.4f}")

if __name__ == "__main__":
    main()