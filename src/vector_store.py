from langchain_chroma import Chroma
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.indexes import index
from langchain.indexes import SQLRecordManager
import os

class VectorStoreManager:
    def __init__(self, embeddings, persist_directory='./chroma_db'):
        self.embeddings = embeddings
        self.persist_directory = persist_directory
        self.vectorstore = None

        # SQLRecordManager の初期化
        self.record_manager = SQLRecordManager(
            namespace="chroma/rag_collection",
            db_url="sqlite:///record_manager_cache.sql"
        )
        self.record_manager.create_schema()

        # Parent Document Retriever の設定
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,  # 親チャンク: 大きめ
            chunk_overlap=200,
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,  # 子チャンク: 小さめ（検索用）
            chunk_overlap=50,
        )
        self.store = InMemoryStore()  # 親ドキュメントの保存用
        self.parent_retriever = None
    
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
        """既存のベクトルストアを読み込み（Parent Document Retriever対応）"""
        if os.path.exists(self.persist_directory):
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )

            # Parent Document Retriever の再初期化
            self.parent_retriever = ParentDocumentRetriever(
                vectorstore=self.vectorstore,
                docstore=self.store,
                child_splitter=self.child_splitter,
                parent_splitter=self.parent_splitter,
            )

            print("既存のベクトルストアを読み込みました")
            print("Parent Document Retriever を再初期化しました")
            return self.vectorstore
        else:
            raise FileNotFoundError("ベクトルストアが見つかりません")
    

    def similarity_search(self, query, k=3, threshold=0.5, filter=None):
        """類似検索（閾値付き、メタデータフィルタリング対応）"""
        if self.vectorstore is None:
            raise ValueError("ベクトルストアが初期化されていません")

        results = self.vectorstore.similarity_search_with_score(query, k=min(k * 2, 10), filter=filter)
        filtered_docs = [
            doc for doc, score in results if score >= threshold
        ]
        final_docs = filtered_docs[:k]

        print(
            f"閾値{threshold}でフィルタリング: {len(results)}件中{len(final_docs)}件を取得"
        )

        return final_docs

        
    def add_documents(self, documents):
        """ベクトルストアにドキュメントを追加（Parent Document Retriever対応）"""
        if self.vectorstore is None:
            raise ValueError("ベクトルストアが初期化されていません")

        # Parent Document Retriever が初期化されている場合はそれを使用
        if self.parent_retriever:
            self.parent_retriever.add_documents(documents)
            print(f"Parent Document Retriever: {len(documents)}件のドキュメントを追加しました")
        else:
            # 通常の追加
            self.vectorstore.add_documents(documents)
            print(f"{len(documents)}件のドキュメントをベクトルストアに追加しました")

    def sync_documents(self, docs_source):
        """差分更新でドキュメントを同期"""
        if self.vectorstore is None:
            raise ValueError("ベクトルストアが初期化されていません")

        result = index(
            docs_source=docs_source,
            record_manager=self.record_manager,
            vector_store=self.vectorstore,
            cleanup="incremental",
            source_id_key="source"
        )

        print(f"ドキュメント同期完了: 追加{result['num_added']}, 更新{result['num_updated']}, スキップ{result['num_skipped']}, 削除{result['num_deleted']}")
        return result

    def get_hybrid_retriever(self, documents=None):
            """ハイブリッド検索（ベクトル + BM25）のRetrieverを取得"""
            from langchain_community.retrievers import BM25Retriever
            from langchain.retrievers import EnsembleRetriever
            from langchain.schema import Document

            if documents is None and self.vectorstore:
                # 既存のドキュメントを取得
                try:
                    all_docs = self.vectorstore.get()
                    docs_texts = all_docs.get('documents', [])
                    metadatas = all_docs.get('metadatas', [])
                    documents = [
                        Document(page_content=text, metadata=meta or {})
                        for text, meta in zip(docs_texts, metadatas)
                    ]
                except Exception as e:
                    print(f"ドキュメント取得エラー: {e}")
                    documents = []

            if not documents:
                print("警告: ハイブリッド検索用のドキュメントがありません")
                return None

            # BM25 Retriever の作成
            bm25_retriever = BM25Retriever.from_documents(documents)
            bm25_retriever.k = 5  # BM25の検索結果数

            # ベクトル検索 Retriever の作成
            vector_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})

            # Ensemble Retriever（ハイブリッド検索）の作成
            ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, vector_retriever],
                weights=[0.4, 0.6]  # BM25: 40%, ベクトル: 60%
            )

            print("ハイブリッド検索 Retriever を作成しました")
            return ensemble_retriever

    def clear_database(self):
        """Chromaデータベースを完全にクリア"""
        import shutil

        # ベクトルストアを削除
        if os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)
            print(f"Chromaデータベースを削除しました: {self.persist_directory}")

        # レコードマネージャーのキャッシュを削除
        if os.path.exists("record_manager_cache.sql"):
            os.remove("record_manager_cache.sql")
            print("レコードマネージャーのキャッシュを削除しました")

        # インスタンス変数をリセット
        self.vectorstore = None
        self.parent_retriever = None

        print("データベースのクリアが完了しました")