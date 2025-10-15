from src.document_loader import DocumentLoader
from src.embeddings import EmbeddingManager
from src.vector_store import VectorStoreManager
from src.aws_utils import BedrockClient
from src.rag_chain import RAGChain
from src.s3_manager import S3Manager
from dotenv import load_dotenv
import os

def setup_demo_documents():
    """デモ用ドキュメントを作成"""
    os.makedirs('documents', exist_ok=True)
    
    # 日本語のサンプルドキュメント
    with open('documents/rag_basics.txt', 'w', encoding='utf-8') as f:
        f.write("""
RAG（Retrieval-Augmented Generation）の基礎

RAGは、大規模言語モデル（LLM）に外部知識を組み合わせる革新的な手法です。
従来のLLMは学習データに含まれる知識のみで回答しますが、RAGでは以下のプロセスで動作します。

1. ユーザーの質問をベクトル化
2. ベクトルデータベースから関連情報を検索
3. 検索結果をコンテキストとしてLLMに渡す
4. LLMが検索結果を元に回答を生成

このアプローチにより、最新情報や特定ドメインの知識を活用できます。
        """)
    
    with open('documents/aws_bedrock.txt', 'w', encoding='utf-8') as f:
        f.write("""
Amazon Bedrockについて

Amazon Bedrockは、AWSが提供するフルマネージド型の生成AIサービスです。
以下の特徴があります：

- Anthropic Claude、Amazon Titan、Meta Llamaなど複数のLLMを利用可能
- API経由で簡単に呼び出し可能
- 従量課金制で初期費用不要
- セキュリティとプライバシーに配慮した設計

料金体系：
- Claude 3 Haiku: 入力1000トークンあたり$0.00025、出力1000トークンあたり$0.00125
- Claude 3.5 Sonnet: より高性能だが高額

東京リージョン(ap-northeast-1)でも利用可能です。
        """)

def main():
    # 環境変数読み込み
    load_dotenv()
    
    print("="*60)
    print("🚀 RAG + AWS Bedrockシステムの起動")
    print("="*60)
    
    # デモドキュメント作成
    setup_demo_documents()
    
    # 1. ドキュメント読み込みとベクトル化
    print("\n📄 ステップ1: ドキュメント処理")
    print("-"*60)
    loader = DocumentLoader()
    documents = loader.load_directory('documents')
    print(f"✅ {len(documents)}個のチャンクを作成")
    
    # 2. 埋め込みモデル準備
    print("\n🤖 ステップ2: 埋め込みモデル準備")
    print("-"*60)
    embedding_manager = EmbeddingManager()
    embeddings = embedding_manager.get_embeddings()
    print("✅ 埋め込みモデル準備完了")
    
    # 3. ベクトルストア構築
    print("\n💾 ステップ3: ベクトルストア構築")
    print("-"*60)
    vector_manager = VectorStoreManager(embeddings)
    
    # 既存のベクトルストアがあれば読み込み、なければ作成
    try:
        vector_manager.load_vectorstore()
    except FileNotFoundError:
        vector_manager.create_vectorstore(documents)
    
    # 4. Bedrock接続テスト
    print("\n🔌 ステップ4: AWS Bedrock接続")
    print("-"*60)
    bedrock_client = BedrockClient(region_name=os.getenv('AWS_REGION', 'ap-northeast-1'))
    
    if not bedrock_client.test_connection():
        print("❌ Bedrock接続に失敗しました。以下を確認してください:")
        print("  1. AWS CLIで認証情報が設定されているか")
        print("  2. Bedrockでモデルアクセスが有効化されているか")
        print("  3. IAMユーザーにbedrock:InvokeModel権限があるか")
        return
    
    # 5. RAGチェーン作成
    print("\n⛓️  ステップ5: RAGチェーン構築")
    print("-"*60)
    rag_chain = RAGChain(vector_manager, bedrock_client)
    print("✅ RAGシステム準備完了")
    
    # 6. 質問応答デモ
    print("\n" + "="*60)
    print("💡 RAGシステムで質問に答えます")
    print("="*60)
    
    questions = [
        "RAGとは何ですか？どのように動作しますか？",
        "Amazon Bedrockの料金体系について教えてください",
        "Claude 3 Haikuの特徴は何ですか？"
    ]
    
    for question in questions:
        result = rag_chain.query(question, k=2)
        rag_chain.pretty_print_result(result)
        print("\n")
    
    # 7. S3連携デモ（オプション）
    print("\n" + "="*60)
    print("☁️  オプション: S3連携デモ")
    print("="*60)
    
    bucket_name = os.getenv('S3_BUCKET_NAME')
    if bucket_name and bucket_name != 'your-rag-documents-bucket':
        try:
            s3_manager = S3Manager(bucket_name)
            s3_manager.create_bucket_if_not_exists()
            
            # ドキュメントをS3にアップロード
            for file in os.listdir('documents'):
                file_path = os.path.join('documents', file)
                s3_manager.upload_file(file_path, f'documents/{file}')
            
            # バケット内のファイル一覧
            files = s3_manager.list_files('documents/')
            print(f"\n📦 S3バケット内のファイル: {files}")
            
        except Exception as e:
            print(f"⚠️  S3連携スキップ: {e}")
    else:
        print("⚠️  S3_BUCKET_NAMEが設定されていないためスキップ")
    
    print("\n" + "="*60)
    print("✅ 全ての処理が完了しました！")
    print("="*60)

if __name__ == "__main__":
    main()