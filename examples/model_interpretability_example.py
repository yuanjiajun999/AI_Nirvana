import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from src.core.model_interpretability import ModelInterpreter

# 加载数据
# 这里使用虚构的数据，实际使用时请替换为您的实际数据
data = pd.DataFrame({
    'feature1': np.random.rand(1000),
    'feature2': np.random.rand(1000),
    'feature3': np.random.rand(1000),
    'target': np.random.choice([0, 1], 1000)
})

X = data.drop('target', axis=1)
y = data['target']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练模型
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 创建ModelInterpreter实例
interpreter = ModelInterpreter(
    model=model,
    X=X,
    y=y,
    feature_names=X.columns.tolist(),
    class_names=['Class 0', 'Class 1'],
    model_type='tree'
)

# 运行所有分析并保存结果
output_dir = 'interpretation_results'
interpreter.run_all_analyses(output_dir)

# 保存模型
model_path = 'model.joblib'
interpreter.save_model(model_path)

# 加载模型
loaded_interpreter = ModelInterpreter(None, X, y)
loaded_interpreter.load_model(model_path)

# 比较原始模型和加载的模型的预测
original_predictions = interpreter.model.predict(X_test)
loaded_predictions = loaded_interpreter.model.predict(X_test)
print("Models produce the same predictions:", np.array_equal(original_predictions, loaded_predictions))