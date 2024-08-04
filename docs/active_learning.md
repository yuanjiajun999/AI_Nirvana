# ActiveLearner 类文档

## 概述

ActiveLearner 类实现了一个主动学习框架，允许用户使用各种采样策略来训练机器学习模型。它支持不确定性采样、多样性采样、预期模型变化采样和密度加权采样等策略。

## 初始化
ActiveLearner(X_pool, y_pool, X_test, y_test, random_state=None)

### 参数

- `X_pool`: 未标记数据的特征矩阵
- `y_pool`: 未标记数据的标签数组
- `X_test`: 测试集的特征矩阵
- `y_test`: 测试集的标签数组
- `random_state`: 随机数生成器的种子（可选）

## 主要方法

### active_learning_loop

执行完整的主动学习循环。
active_learning_loop(initial_samples, n_iterations, samples_per_iteration, strategy='uncertainty', batch_mode=False)

#### 参数

- `initial_samples`: 初始标记样本的数量
- `n_iterations`: 主动学习循环的迭代次数
- `samples_per_iteration`: 每次迭代选择的样本数
- `strategy`: 采样策略，可选 'uncertainty', 'diversity', 'expected_model_change', 'density_weighted'
- `batch_mode`: 是否使用批处理模式

#### 返回值

- `final_accuracy`: 最终模型准确率
- `accuracy_history`: 每次迭代后的准确率列表

### train

训练模型。
train(X, y)

#### 参数

- `X`: 训练数据的特征矩阵
- `y`: 训练数据的标签数组

### evaluate

评估模型性能。
evaluate(metric='accuracy')

#### 参数

- `metric`: 评估指标，可选 'accuracy', 'f1', 'precision', 'recall'

#### 返回值

- 选定指标的评分

### uncertainty_sampling

执行不确定性采样。
uncertainty_sampling(n_samples)

#### 参数

- `n_samples`: 要选择的样本数

#### 返回值

- 选定样本的索引数组

### diversity_sampling

执行多样性采样。
diversity_sampling(n_samples)

#### 参数

- `n_samples`: 要选择的样本数

#### 返回值

- 选定样本的索引数组

### expected_model_change_sampling

执行预期模型变化采样。
expected_model_change_sampling(n_samples)

#### 参数

- `n_samples`: 要选择的样本数

#### 返回值

- 选定样本的索引数组

### density_weighted_sampling

执行密度加权采样。
density_weighted_sampling(n_samples)

#### 参数

- `n_samples`: 要选择的样本数

#### 返回值

- 选定样本的索引数组

### create_committee

创建模型委员会。
create_committee(n_models=3)

#### 参数

- `n_models`: 委员会中的模型数量

### committee_uncertainty_sampling

使用委员会进行不确定性采样。
committee_uncertainty_sampling(n_samples)

#### 参数

- `n_samples`: 要选择的样本数

#### 返回值

- 选定样本的索引数组

### plot_learning_curve

绘制学习曲线。
plot_learning_curve(accuracy_history)

#### 参数

- `accuracy_history`: 准确率历史记录列表

## 使用示例

请参考上面的示例代码，了解如何使用 ActiveLearner 类的各种功能。

## 注意事项

1. 在使用 `committee_uncertainty_sampling` 之前，请确保已经调用了 `create_committee`。
2. 不同的采样策略可能适用于不同的场景，建议尝试多种策略并比较结果。
3. 主动学习的效果可能会因数据集的特性而异，请根据具体情况调整参数。