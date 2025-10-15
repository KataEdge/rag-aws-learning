from src.document_loader import DocumentLoader
from src.embeddings import EmbeddingManager
from src.vector_store import VectorStoreManager
from dotenv import load_dotenv
import os

def main():
    # 環境変数読み込み
    load_dotenv()
    
    # 1. ドキュメント読み込み
    print("📄 ドキュメントを読み込み中...")
    loader = DocumentLoader()
    
    # テスト用ドキュメント作成
    os.makedirs('documents', exist_ok=True)
    with open('documents/test.txt', 'w', encoding='utf-8') as f:
        f.write("""
        RAG（Retrieval-Augmented Generation）は、大規模言語モデルに外部知識を組み合わせる手法です。
        ベクトルデータベースを使って関連情報を検索し、その情報を元にLLMが回答を生成します。
        AWSではAmazon BedrockとOpenSearch Serverlessを組み合わせてRAGシステムを構築できます。
        """)
    
    documents = loader.load_directory('documents')
    print(f"✅ {len(documents)}個のチャンクを作成しました")
    
    # 2. 埋め込みモデル初期化
    print("\n🤖 埋め込みモデルを読み込み中...")
    embedding_manager = EmbeddingManager()
    embeddings = embedding_manager.get_embeddings()
    print("✅ 埋め込みモデル準備完了")
    
    # 3. ベクトルストア作成
    print("\n💾 ベクトルストアを作成中...")
    vector_manager = VectorStoreManager(embeddings)
    vectorstore = vector_manager.create_vectorstore(documents)
    print("✅ ベクトルストア作成完了")
    
    # 4. 検索テスト
    print("\n🔍 検索テスト実行中...")
    query = "RAGとは何ですか？"
    results = vector_manager.similarity_search(query, k=2)
    
    print(f"\n質問: {query}")
    print("\n検索結果:")
    for i, doc in enumerate(results, 1):
        print(f"\n--- 結果 {i} ---")
        print(doc.page_content)

if __name__ == "__main__":
    main()