from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertModel, BertTokenizer

from .model_interface import ModelInterface


class LocalModel(ModelInterface):
    def __init__(self):
        self.model_name = "bert-base-chinese"
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertModel.from_pretrained(self.model_name)
        self.responses = {
            "自我介绍": "我是AI Nirvana智能助手，一个基于人工智能技术的对话系统。我可以回答问题、提供信息，并尽力协助您完成各种任务。虽然我的知识有限，但我会不断学习和改进。",
            "你是谁": "我是AI Nirvana智能助手，一个AI对话系统。我的目标是协助用户并回答问题。",
            "你能做什么": "作为AI助手，我可以回答问题、提供信息、进行简单的计算和文本分析。但请记住，我的能力有限，无法执行物理任务或访问外部信息。",
            "生成式AI": "生成式AI是人工智能领域的一个重要分支，它能够创造新的内容，如文本、图像、音频等。目前，生成式AI发展迅速，已在多个领域展现出巨大潜力，如自然语言处理、计算机视觉等。未来，生成式AI有望在创意产业、教育、医疗等方面发挥更大作用，但也面临着伦理和安全等挑战。",
            "AI发展": "AI技术正在快速发展，影响着各个行业。主要趋势包括深度学习、强化学习、自然语言处理等领域的突破。未来AI可能在医疗诊断、自动驾驶、个人助理等方面带来革命性变化。同时，AI伦理、隐私保护等问题也日益重要。",
            "默认回答": "对不起，我没有足够的信息来回答这个问题。您能提供更多细节或换个话题吗？"
        }
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.responses.keys())

    def generate_response(self, prompt):
        query_vec = self.vectorizer.transform([prompt])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        best_match_index = similarities.argmax()
        best_match_key = list(self.responses.keys())[best_match_index]
        
        if similarities[best_match_index] < 0.1:
            return self.responses["默认回答"]
        
        return self.responses[best_match_key]

    def summarize(self, text):
        return text[:100] + "..."  # 简单地返回文本的前100个字符作为"摘要"