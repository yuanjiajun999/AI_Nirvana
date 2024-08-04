import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from src.core.active_learning import ActiveLearner

# 生成示例数据
X, y = make_classification(n_samples=1000, n_features=20, n_informative=2,
                           n_redundant=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化 ActiveLearner
learner = ActiveLearner(X_train, y_train, X_test, y_test, random_state=42)
learner.model = RandomForestClassifier(random_state=42)

# 执行主动学习循环
initial_samples = 50
n_iterations = 10
samples_per_iteration = 10

final_accuracy, accuracy_history = learner.active_learning_loop(
    initial_samples=initial_samples,
    n_iterations=n_iterations,
    samples_per_iteration=samples_per_iteration,
    strategy='uncertainty'
)

print(f"Final accuracy: {final_accuracy:.4f}")

# 使用不同的采样策略
strategies = ['uncertainty', 'diversity', 'expected_model_change', 'density_weighted']
for strategy in strategies:
    print(f"\nTesting {strategy} strategy:")
    accuracy, _ = learner.active_learning_loop(
        initial_samples=50,
        n_iterations=5,
        samples_per_iteration=10,
        strategy=strategy
    )
    print(f"{strategy} strategy final accuracy: {accuracy:.4f}")

# 使用委员会进行采样
learner.create_committee(n_models=3)
committee_indices = learner.committee_uncertainty_sampling(10)
print(f"\nCommittee selected indices: {committee_indices}")

# 绘制学习曲线
learner.plot_learning_curve(accuracy_history)

# 手动执行一次采样和训练
selected_indices = learner.uncertainty_sampling(10)
learner.label_samples(selected_indices)
X_labeled, y_labeled = learner.get_labeled_data()
learner.train(X_labeled, y_labeled)

# 评估模型
accuracy = learner.evaluate()
print(f"\nAccuracy after manual sampling and training: {accuracy:.4f}")

# 使用不同的评估指标
metrics = ['accuracy', 'f1', 'precision', 'recall']
for metric in metrics:
    score = learner.evaluate(metric=metric)
    print(f"{metric.capitalize()} score: {score:.4f}")