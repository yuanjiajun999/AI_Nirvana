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


# AI Nirvana 主动学习系统文档

## 1. 系统概述

AI Nirvana 主动学习系统是一个强大的工具，用于实现和管理主动学习过程。它允许用户初始化学习器、标记数据、创建和管理模型委员会，以及执行各种相关任务。

## 2. 主要功能

### 2.1 初始化主动学习器
命令: `init_active_learner`
描述: 初始化主动学习系统，加载必要的数据。

### 2.2 标记初始数据
命令: `label_initial_data`
描述: 标记一定数量的初始样本，为主动学习过程做准备。

### 2.3 管理主动学习模型
命令: `al_model`
描述: 获取或设置当前的主动学习模型。

### 2.4 创建模型委员会
命令: `al_committee`
描述: 创建一个包含多个模型的委员会，用于主动学习。

### 2.5 执行主动学习循环
命令: `run_active_learning`
描述: 执行主动学习的迭代过程。

### 2.6 可视化学习曲线
命令: `al_plot`
描述: 绘制主动学习的性能曲线。

## 3. 使用指南

### 3.1 初始化系统
1. 运行 `init_active_learner` 命令初始化系统。
2. 使用 `label_initial_data` 命令标记一些初始样本。

### 3.2 设置和查看模型
1. 使用 `al_model` 命令，选择 'get' 选项查看当前模型。
2. 使用 `al_model` 命令，选择 'set' 选项设置新模型。
   可选模型包括：Random Forest, Support Vector Machine, Logistic Regression。

### 3.3 创建模型委员会
使用 `al_committee` 命令，指定委员会中模型的数量。

### 3.4 执行主动学习
使用 `run_active_learning` 命令开始主动学习过程。
可以指定迭代次数和每次迭代的样本数。

### 3.5 可视化结果
使用 `al_plot` 命令查看学习曲线，了解模型性能随时间的变化。

## 4. 注意事项

- 在使用其他功能之前，请确保已经初始化了主动学习器并标记了一些初始数据。
- 更改模型后，可能需要重新创建委员会以确保一致性。
- 主动学习过程可能需要较长时间，请确保系统有足够的资源。

## 5. 错误处理

- 如果遇到 "Active learner not initialized" 错误，请使用 `init_active_learner` 命令。
- 如果遇到 "No labeled data available" 错误，请使用 `label_initial_data` 命令标记一些数据。

## 6. 性能优化建议

- 从小数据集开始，逐步增加数据量。
- 定期评估模型性能，关注学习曲线的变化。
- 尝试不同的模型和委员会配置，找出最适合您数据的设置。

## 7. 未来改进方向

- 添加更多的模型选项。
- 实现自动化的超参数调整。
- 增加更多的可视化选项。

如有任何问题或需要进一步的帮助，请随时查阅此文档或联系系统管理员。