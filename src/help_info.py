from src.command_data import COMMAND_DESCRIPTIONS

def get_help(topic=None):
    if topic is None:
        return "\n".join([f"{cmd}: {desc}" for cmd, desc in COMMAND_DESCRIPTIONS.items()])
    elif topic in COMMAND_DESCRIPTIONS:
        return f"{topic}: {COMMAND_DESCRIPTIONS[topic]}"
    else:
        return f"未找到关于 '{topic}' 的帮助信息。"

def general_help():
    help_text = "可用命令：\n"
    for cmd, desc in COMMAND_DESCRIPTIONS.items():
        help_text += f"'{cmd}' - {desc}\n"
    
    help_text += "\n注意事项：\n"
    help_text += "- 执行代码时，某些操作（如文件操作和模块导入）出于安全考虑是受限的。\n"
    help_text += "- 支持基本的Python操作，包括变量赋值、条件语句、循环等。\n"
    help_text += "- 如果遇到'未定义'的错误，可能是因为该操作被安全限制所阻止。\n"
    # ... 添加其他所有注意事项 ...
    help_text += "- API调用可能会受限于API的速率限制和配额，请注意控制调用频率以避免超出限制。\n"
    
    return help_text

def get_feature_engineering_help():
    return """
特征工程相关命令：
- 'init_feature_engineer' - 初始化特征工程器
  - 支持使用示例数据或本地文件
  - 使用示例数据时可以指定样本数量和特征数量
  - 使用本地文件时需要提供文件路径和目标列名称
- 'get_important_features' - 获取最重要的特征（需要先初始化特征工程器）
"""

def get_digital_twin_help():
    return """
数字孪生系统相关注意事项：
- 在使用 `create_digital_twin` 命令时，请确保物理系统模型函数定义正确且符合要求。
- 在进行模拟、监控和优化操作前，请确认数字孪生系统已经成功创建。
- 使用 `update_digital_twin_model` 更新物理模型时，请谨慎操作，确保新模型的准确性和稳定性。
- 进行大规模模拟和优化任务时，请注意系统资源的使用，避免因资源不足导致的性能问题。
- 在处理复杂的物理系统模型时，建议进行充分的测试，以确保系统行为符合预期。
- 数字孪生系统的各项功能依赖于输入数据的准确性，请确保输入的初始条件、时间步长、传感器数据等信息的准确性。
"""

def get_model_management_help():
    return """
模型管理相关注意事项：
- 'generate_text' 命令用于生成文本，可以指定最大令牌数和温度参数。
- 'classify_image' 和 'caption_image' 命令需要提供有效的图像文件路径。
- 'fine_tune_model' 命令用于微调模型，需要提供训练数据和相关参数。
- 使用 'save_model' 和 'load_model' 命令时，请确保指定正确的文件路径。
- 图像处理和模型微调功能可能需要较长时间，请耐心等待。
- 在使用模型相关功能时，请确保系统有足够的计算资源。
- 'save_model' 和 'load_model' 命令现在可以处理both通用模型和智能代理模型。使用时请指定模型类型（'general' 或 'agent'）。
"""

def get_agent_help():
    return """
智能代理相关注意事项：
- 在使用智能代理相关功能（如 'create_agent'、'train_agent'）之前，请确保已安装必要的依赖，如 TensorFlow 和 OpenAI Gym。
- 智能代理的训练可能需要较长时间，请耐心等待。训练过程中会显示进度信息。
- 在使用 'rl_decide' 命令时，请确保先使用 'setup_rl_agent' 设置了强化学习代理。
"""

def get_langgraph_help():
    return """
LangGraph相关注意事项：
- LangGraph支持对实体的添加、更新和删除操作。请确保在添加实体时使用唯一的名称，以避免覆盖已有的实体。
- 在更新实体信息时，新属性将覆盖已有的属性。如果只想修改部分属性，请确保只提供需要更新的键值对。
- 可以为两个实体添加关系。关系的名称应当具有描述性，以便清晰表达实体之间的关联。删除实体时，该实体的所有关系也将被自动删除。
- LangGraph支持基于知识图的常识推理和语义搜索功能。这些功能依赖于知识图中的实体和关系的正确配置。请在进行复杂推理之前，确保图结构的完整性和准确性。
- 语义搜索功能可以通过相似度匹配返回相关的实体。搜索结果依赖于嵌入向量的质量，因此建议在关键实体被添加或更新后重新构建嵌入。
- 在查询时，建议使用清晰且简洁的语言，以获得更准确的匹配结果。
- 导出知识图时，可以选择不同的格式（如GraphML或GEXF）。这些格式适合在外部工具中进行可视化和进一步分析。
- 知识图的摘要功能提供了图的基本统计信息，如节点和边的数量。这有助于了解图的整体结构和复杂度。
- LangGraph模块依赖于OpenAI API和嵌入模型。在使用该模块前，请确保正确配置API密钥和模型参数。
- API调用可能会受限于API的速率限制和配额，请注意控制调用频率以避免超出限制。
"""