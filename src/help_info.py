def print_help() -> None:
    print("\n可用命令：")
    descriptions = {
        'quit': '退出程序',
        'clear': '清除对话历史和变量状态',
        'sentiment': '对下一条输入进行情感分析',
        'execute': '执行 Python 代码',
        'validate_code': '验证 Python 代码的安全性',
        'supported_languages': '显示支持的编程语言列表',
        'summarize': '生成文本摘要',
        'change_model': '更改使用的模型',
        'vars': '显示当前定义的所有变量',
        'explain': '解释模型预测',
        'encrypt': '加密敏感数据',
        'decrypt': '解密数据',
        'add_knowledge': '添加知识到知识库',
        'get_knowledge': '检索知识库中的知识',
        'list_knowledge': '列出所有知识库中的知识',
        'update_knowledge': '更新知识库中的知识',
        'delete_knowledge': '从知识库中删除知识',
        'search_knowledge': '在知识库中搜索知识',
        'extract_keywords': '从文本中提取关键词',
        'plan_task': '为给定任务生成计划',
        'translate': '翻译文本到指定语言',
        'help': '显示此帮助信息',
        'test_complex': '执行复杂文本分析测试',
        'test_models': '执行模型比较测试',
        'test_code_safety': '执行代码安全性测试',
        'qa': '问答功能',
        'init_active_learner': '初始化主动学习器',
        'run_active_learning': '执行主动学习循环',
        'al_model': '获取或设置主动学习模型',
        'al_committee': '创建主动学习委员会',
        'al_plot': '绘制主动学习曲线',
        'label_initial_data': '标记数据',
        'view_committee': '再次查看委员会',
        'init_feature_engineer': '初始化特征工程器',
        'create_entity_set': '创建实体集',
        'generate_features': '生成特征',
        'get_important_features': '获取重要特征',
        'remove_low_info_features': '移除低信息特征',
        'remove_correlated_features': '移除高度相关特征',
        'create_custom_feature': '创建自定义特征',
        'get_feature_types': '获取特征类型',
        'get_feature_descriptions': '获取特征描述',
        'normalize_features': '标准化特征',
        'encode_categorical_features': '编码分类特征',
        'create_digital_twin': '创建一个新的数字孪生系统',
        'simulate_digital_twin': '模拟数字孪生系统状态变化',
        'monitor_digital_twin': '监控数字孪生系统状态并检测异常',
        'optimize_digital_twin': '优化数字孪生系统参数',
        'update_digital_twin_model': '更新数字孪生系统的物理模型',
        'validate_digital_twin_model': '验证数字孪生系统模型的准确性',
        'generate_text': '生成文本',
        'classify_image': '对图像进行分类',
        'caption_image': '为图像生成描述',
        'fine_tune_model': '微调模型',
        'save_model': '保存模型（通用模型或智能代理模型)',
        'load_model': '加载模型（通用模型或智能代理模型)',
        'create_agent' : '加 创建一个新的 DQN 智能代理',
        'train_agent' : '在指定环境中训练智能代理',
        'run_agent' : '在指定环境中运行智能代理',
        'setup_rl_agent' : '设置强化学习代理',
        'rl_decide' : '让强化学习代理根据给定状态做出决策',
        'add_entity': '添加实体到知识图',
        'update_entity': '更新实体信息',
        'delete_entity': '删除实体',
        'add_relation': '添加实体之间的关系',
        'get_graph_summary': '获取知识图的摘要',
        'export_graph': '导出知识图',
        'infer_commonsense': '执行常识推理',
        'retrieve_knowledge': '检索知识图中的知识',
        'semantic_search': '执行语义搜索',
        'get_entity_info': '获取特定实体的信息',
        'get_related_entities': '获取与特定实体相关的实体',
        'get_all_entities': '获取所有实体',
        'get_all_relationships': '获取所有关系',
    }

    for cmd, desc in descriptions.items():
        print(f"'{cmd}' - {desc}")

    print("\n注意：")
    print("- 执行代码时，某些操作（如文件操作和模块导入）出于安全考虑是受限的。")
    print("- 支持基本的Python操作，包括变量赋值、条件语句、循环等。")
    print("- 如果遇到'未定义'的错误，可能是因为该操作被安全限制所阻止。")
    print("- 请按正确顺序使用主动学习相关命令：先初始化主动学习器，再执行主动学习。")
    print("- 主动学习可能涉及大量数据和计算，请确保有足够的系统资源。")
    print("- 长时间运行的操作可以通过 Ctrl+C 中断。")
    print("'init_feature_engineer' - 初始化特征工程器")
    print("  - 支持使用示例数据或本地文件")
    print("  - 使用示例数据时可以指定样本数量和特征数量")
    print("  - 使用本地文件时需要提供文件路径和目标列名称")
    print("'get_important_features' - 获取最重要的特征（需要先初始化特征工程器）")
    print("- 在使用 `create_digital_twin` 命令时，请确保物理系统模型函数定义正确且符合要求。")
    print("- 在进行模拟、监控和优化操作前，请确认数字孪生系统已经成功创建。")
    print("- 使用 `update_digital_twin_model` 更新物理模型时，请谨慎操作，确保新模型的准确性和稳定性。")
    print("- 进行大规模模拟和优化任务时，请注意系统资源的使用，避免因资源不足导致的性能问题。")
    print("- 在处理复杂的物理系统模型时，建议进行充分的测试，以确保系统行为符合预期。")
    print("- 数字孪生系统的各项功能依赖于输入数据的准确性，请确保输入的初始条件、时间步长、传感器数据等信息的准确性。")
    print("- 'generate_text' 命令用于生成文本，可以指定最大令牌数和温度参数。")
    print("- 'classify_image' 和 'caption_image' 命令需要提供有效的图像文件路径。")
    print("- 'fine_tune_model' 命令用于微调模型，需要提供训练数据和相关参数。")
    print("- 使用 'save_model' 和 'load_model' 命令时，请确保指定正确的文件路径。")
    print("- 图像处理和模型微调功能可能需要较长时间，请耐心等待。")
    print("- 在使用模型相关功能时，请确保系统有足够的计算资源。")
    print("- 'save_model' 和 'load_model' 命令现在可以处理both通用模型和智能代理模型。使用时请指定模型类型（'general' 或 'agent'）。")
    print("- 在使用智能代理相关功能（如 'create_agent'、'train_agent'）之前，请确保已安装必要的依赖，如 TensorFlow 和 OpenAI Gym。")
    print("- 智能代理的训练可能需要较长时间，请耐心等待。训练过程中会显示进度信息。")
    print("- 在使用 'rl_decide' 命令时，请确保先使用 'setup_rl_agent' 设置了强化学习代理。")
    print("- 如果遇到任何未预期的错误或异常行为，请检查日志文件以获取更详细的信息。")
    print("- 定期保存您的工作成果和模型，以防意外情况发生。")
    print("- 如果您是在共享环境中使用本系统，请注意保护敏感数据和模型的安全。")
    print("- LangGraph支持对实体的添加、更新和删除操作。请确保在添加实体时使用唯一的名称，以避免覆盖已有的实体。")
    print("- 在更新实体信息时，新属性将覆盖已有的属性。如果只想修改部分属性，请确保只提供需要更新的键值对。")
    print("- 可以为两个实体添加关系。关系的名称应当具有描述性，以便清晰表达实体之间的关联。删除实体时，该实体的所有关系也将被自动删除。")
    print("- LangGraph支持基于知识图的常识推理和语义搜索功能。这些功能依赖于知识图中的实体和关系的正确配置。请在进行复杂推理之前，确保图结构的完整性和准确性。")
    print("- 语义搜索功能可以通过相似度匹配返回相关的实体。搜索结果依赖于嵌入向量的质量，因此建议在关键实体被添加或更新后重新构建嵌入。")
    print("- 在查询时，建议使用清晰且简洁的语言，以获得更准确的匹配结果。")
    print("- 导出知识图时，可以选择不同的格式（如GraphML或GEXF）。这些格式适合在外部工具中进行可视化和进一步分析。")
    print("- 知识图的摘要功能提供了图的基本统计信息，如节点和边的数量。这有助于了解图的整体结构和复杂度。")
    print("- LangGraph模块依赖于OpenAI API和嵌入模型。在使用该模块前，请确保正确配置API密钥和模型参数。")
    print("- API调用可能会受限于API的速率限制和配额，请注意控制调用频率以避免超出限制。")