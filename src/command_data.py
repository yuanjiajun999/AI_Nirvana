AVAILABLE_COMMANDS = [
    'quit', 'clear', 'sentiment', 'execute', 'validate_code', 
    'supported_languages', 'summarize', 'change_model', 
    'vars', 'explain', 'encrypt', 'decrypt', 'kb_add', 'kb_query', 'kb_update', 
    'kb_delete', 'kb_list' 'extract_keywords', 
    'plan_task', 'translate', 'help', 'test_complex', 'test_models', 
    'test_code_safety', 'qa', 'init_active_learner', 'run_active_learning', 
    'al_model', 'al_committee', 'al_plot', 'label_initial_data', 
    'view_committee', 'init_feature_engineer', 'create_entity_set', 
    'generate_features', 'get_important_features', 'remove_low_info_features',
    'remove_correlated_features', 'create_custom_feature', 'get_feature_types', 
    'get_feature_descriptions', 'normalize_features', 'encode_categorical_features',
    'create_digital_twin', 'simulate_digital_twin', 'monitor_digital_twin',
    'optimize_digital_twin', 'update_digital_twin_model', 'validate_digital_twin_model', 
    'generate_text', 'classify_image', 'caption_image', 'fine_tune_model', 
    'save_model', 'load_model', 'create_agent', 'train_agent', 'run_agent', 
    'setup_rl_agent', 'rl_decide', 'add_entity', 'update_entity', 'delete_entity', 
    'add_relation', 'get_graph_summary', 'export_graph', 'infer_commonsense',
    'retrieve_knowledge', 'semantic_search', 'get_entity_info', 'get_related_entities', 
    'get_all_entities', 'get_all_relationships', 'query_knowledge'
]

COMMAND_DESCRIPTIONS = {
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
    'kb_add': '向知识库添加新条目',  
    'kb_query': '查询知识库',  
    'kb_update': '更新知识库中的条目',  
    'kb_delete': '从知识库中删除条目',  
    'kb_list': '列出知识库中的所有条目',  
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
    'create_agent': '创建一个新的 DQN 智能代理',
    'train_agent': '在指定环境中训练智能代理',
    'run_agent': '在指定环境中运行智能代理',
    'setup_rl_agent': '设置强化学习代理',
    'rl_decide': '让强化学习代理根据给定状态做出决策',
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
    'query_knowledge': '直接查询知识库'
}