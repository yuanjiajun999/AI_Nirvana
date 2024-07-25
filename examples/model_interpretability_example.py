# examples/model_interpretability_example.py

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

from src.core.model_interpretability import ModelInterpreter


def main():
    # 加载示例数据集
    iris = load_iris()
    X, y = iris.data, iris.target

    # 训练一个简单的模型
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)

    # 初始化模型解释器
    interpreter = ModelInterpreter(model, X, feature_names=iris.feature_names)

    # 特征重要性
    feature_importance = interpreter.feature_importance()
    print("Feature Importance:")
    for feature, importance in feature_importance.items():
        print(f"{feature}: {importance:.4f}")
    print()

    # 部分依赖图
    feature = "petal length (cm)"
    pdp = interpreter.partial_dependence_plot(feature)
    print(f"Partial Dependence Plot for {feature}:")
    print(pdp)
    print()

    # SHAP值
    shap_values = interpreter.shap_values()
    print("SHAP Values (first 5 samples):")
    print(shap_values[:5])


if __name__ == "__main__":
    main()
