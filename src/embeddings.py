from langchain_community.embeddings import HuggingFaceEmbeddings
import torch

class EmbeddingManager:
    def __init__(self, model_name='BAAI/bge-m3'):
        """
        多言語対応の埋め込みモデル（日本語もサポート）
        """
        # デバイスの自動選択
        if torch.backends.mps.is_available():
            device = 'mps'  # M1/M2/M3 Mac
            print("🚀 MPS (GPU) を使用します - 高速化モード")
        else:
            device = 'cpu'  # Intel Mac or MPS非対応
            print("💻 CPU を使用します")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}
        )
    
    def get_embeddings(self):
        return self.embeddings