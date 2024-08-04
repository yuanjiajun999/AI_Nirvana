# 模型解释性模块使用文档

## 简介

模型解释性模块提供了一套全面的工具，用于解释和可视化机器学习模型的行为。这个模块封装在`ModelInterpreter`类中，支持多种解释技术，包括SHAP值、LIME、特征重要性、混淆矩阵等。

## 安装

确保您已经安装了所有必要的依赖：
pip install pandas numpy scikit-learn shap lime matplotlib seaborn

## 使用方法

### 1. 导入必要的库

```python
from src.core.model_interpretability import ModelInterpreter
2. 准备数据和模型
在创建ModelInterpreter实例之前，您需要准备好您的数据和训练好的模型。
3. 创建ModelInterpreter实例
interpreter = ModelInterpreter(
    model=your_trained_model,
    X=your_feature_data,
    y=your_target_data,
    feature_names=your_feature_names,
    class_names=your_class_names,
    model_type='tree'  # 或 'linear'，取决于您的模型类型
)
4. 运行分析
您可以运行所有分析并将结果保存到指定目录：
output_dir = 'interpretation_results'
interpreter.run_all_analyses(output_dir)
这将生成以下分析结果：

SHAP摘要图
SHAP力图
LIME解释
特征重要性图
混淆矩阵
ROC曲线
学习曲线
排列重要性

5. 保存和加载模型
保存模型：
model_path = 'model.joblib'
interpreter.save_model(model_path)
加载模型：
loaded_interpreter = ModelInterpreter(None, X, y)
loaded_interpreter.load_model(model_path)
主要方法说明

run_all_analyses(output_dir): 运行所有分析并保存结果。
save_model(model_path): 保存模型到指定路径。
load_model(model_path): 从指定路径加载模型。
plot_shap_summary(): 绘制SHAP摘要图。
plot_shap_force(): 绘制SHAP力图。
explain_instance_with_lime(instance_index): 使用LIME解释特定实例。
plot_feature_importance(): 绘制特征重要性图。
plot_confusion_matrix(): 绘制混淆矩阵。
plot_roc_curve(): 绘制ROC曲线。
plot_learning_curve(): 绘制学习曲线。
plot_permutation_importance(): 绘制排列重要性。

注意事项

确保为不同的模型类型（'tree'或'linear'）选择正确的model_type参数。
某些分析方法可能需要较长时间来执行，特别是对于大型数据集。
所有生成的图表都会自动保存到指定的输出目录。
如果遇到中文显示问题，可能需要设置合适的中文字体。

示例
请参考example_model_interpretability.py文件，其中包含了使用此模块的完整示例。
故障排除
如果遇到任何问题，请检查以下几点：

确保所有依赖库都已正确安装。
检查数据格式是否正确（DataFrame对于特征，Series对于目标变量）。
确保模型类型与model_type参数匹配。
如果出现内存错误，尝试使用较小的数据子集。

如果问题仍然存在，请查看完整的错误消息并搜索相关解决方案。
