import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("FAISS not available, falling back to basic vector store")

class VectorStore:
    def __init__(self, dimension):
        self.dimension = dimension
        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatL2(dimension)
        else:
            self.vectors = []
        self.texts = []

    def add_vectors(self, vectors, texts):
        if FAISS_AVAILABLE:
            self.index.add(np.array(vectors).astype('float32'))
        else:
            self.vectors.extend(vectors)
        self.texts.extend(texts)

    def search(self, query_vector, k=5):
        if FAISS_AVAILABLE:
            D, I = self.index.search(np.array([query_vector]).astype('float32'), k)
            return [(self.texts[i], d) for i, d in zip(I[0], D[0])]
        else:
            # 简单的线性搜索实现
            distances = [np.linalg.norm(np.array(query_vector) - np.array(v)) for v in self.vectors]
            sorted_indices = np.argsort(distances)[:k]
            return [(self.texts[i], distances[i]) for i in sorted_indices]