from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from core.active_learning import ActiveLearner
from core.auto_feature_engineering import AutoFeatureEngineer
from core.model_interpretability import ModelInterpreter
from core.reinforcement_learning import ReinforcementLearningAgent

# 创建示例数据
X, y = make_classification(
    n_samples=1000, n_features=20, n_informative=10, n_classes=2, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 强化学习示例
rl_agent = ReinforcementLearningAgent(state_size=20, action_size=2)
state = X_train[0]
action = rl_agent.act(state)
print(f"RL Agent Action: {action}")

# 自动特征工程示例
df = pd.DataFrame(X_train, columns=[f"feature_{i}" for i in range(20)])
df["id"] = range(len(df))
df["timestamp"] = pd.date_range(start="1/1/2021", periods=len(df))
auto_fe = AutoFeatureEngineer(df)
auto_fe.create_entity_set()
feature_matrix, feature_defs = auto_fe.generate_features()
print(f"Generated features: {auto_fe.get_important_features(5)}")

# 模型解释性示例

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
interpreter = ModelInterpreter(model, pd.DataFrame(X_train))
interpreter.create_explainer()
interpreter.plot_summary()

# 主动学习示例
active_learner = ActiveLearner(X_train, y_train, X_test, y_test)
final_accuracy = active_learner.active_learning_loop(
    initial_samples=100, n_iterations=5, samples_per_iteration=20
)
print(f"Final accuracy after active learning: {final_accuracy}")
