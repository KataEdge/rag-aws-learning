from langchain.vectorstores import Chroma
import os

class VectorStoreManager:
    def __init__(self, embeddings, persist_directory='./chroma_db'):
        self.embeddings = embeddings
        self.persist_directory = persist_directory
        self.vectorstore = None
    
    def create_vectorstore(self, documents):
        """新規ベクトルストア作成"""
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        print(f"ベクトルストアを作成しました: {len(documents)}件のドキュメント")
        return self.vectorstore
    
    def load_vectorstore(self):
        """既存のベクトルストアを読み込み"""
        if os.path.exists(self.persist_directory):
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            print("既存のベクトルストアを読み込みました")
            return self.vectorstore
        else:
            raise FileNotFoundError("ベクトルストアが見つかりません")
    
    def similarity_search(self, query, k=3):
        """類似検索"""
        if self.vectorstore is None:
            raise ValueError("ベクトルストアが初期化されていません")
        return self.vectorstore.similarity_search(query, k=k)