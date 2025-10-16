import gradio as gr
from src.document_loader import DocumentLoader
from src.embeddings import EmbeddingManager
from src.vector_store import VectorStoreManager
from src.aws_utils import BedrockClient
from src.rag_chain import RAGChain
from src.s3_manager import S3Manager
from dotenv import load_dotenv
import os

# ---
# グローバル変数としてRAGチェーンを保持
# ---
rag_chain = None

def setup_documents_directory():
    """documentsディレクトリを作成し、ユーザーにファイル配置を促す"""
    os.makedirs('documents', exist_ok=True)
    print("ℹ️  'documents' ディレクトリを準備しました。")
    print("ここに情報源としたいPDFやテキストファイルを置いてください。")
    return True # 常に成功を返す

import shutil

def initialize_rag_system():
    """RAGシステムを初期化"""
    global rag_chain
    
    # 環境変数読み込み
    load_dotenv()
    
    print("="*60)
    print("🚀 RAG + AWS Bedrockシステムの起動")
    print("="*60)
    
    # ドキュメントディレクトリを準備
    if not setup_documents_directory():
        # この分岐は現在通りませんが、念のため残しておきます
        print("ドキュメントディレクトリの準備に失敗しました。")
        return False, "ディレクトリ準備失敗"
        
    # 古いベクトルストアを削除
    if os.path.exists('chroma'):
        print("🗑️  古いベクトルストアを削除しています...")
        shutil.rmtree('chroma')
    
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
    
    # 常にベクトルストアを再構築
    print("🔄 ベクトルストアを再構築します...")
    vector_manager.create_vectorstore(documents)
    
    # 4. Bedrock接続テスト
    print("\n🔌 ステップ4: AWS Bedrock接続")
    print("-"*60)
    bedrock_client = BedrockClient(region_name=os.getenv('AWS_REGION', 'ap-northeast-1'))
    
    if not bedrock_client.test_connection():
        print("❌ Bedrock接続に失敗しました。")
        return False, "Bedrock接続に失敗"
    
    # 5. RAGチェーン作成
    print("\n⛓️  ステップ5: RAGチェーン構築")
    print("-"*60)
    rag_chain = RAGChain(vector_manager, bedrock_client)
    print("✅ RAGシステム準備完了")
    return True, "RAGシステム準備完了"

def answer_question(question):
    """質問に回答する"""
    if rag_chain is None:
        return "RAGシステムが初期化されていません。", "ソースはありません。"
    
    print(f"\n🤔 質問受信: {question}")
    result = rag_chain.query(question, k=2)
    
    # 回答とソースを整形
    answer = result.get('answer', "回答が見つかりませんでした。")
    
    source_documents = result.get('source_documents', [])
    sources_text = "\n\n---\n\n**ソース:**\n"
    if source_documents:
        for doc in source_documents:
            source_info = doc.metadata.get('source', '不明なソース')
            sources_text += f"- {source_info}\n"
    else:
        sources_text += "関連するソースは見つかりませんでした。"
    
    print("✅ 回答生成完了")
    return answer, sources_text

def main():
    # RAGシステムの初期化
    success, message = initialize_rag_system()
    if not success:
        print(message)
        return
        
    # Gradio UIの構築
    with gr.Blocks(title="AWS Bedrock RAG Demo") as demo:
        gr.Markdown("# AWS Bedrockを使ったRAGシステム")
        gr.Markdown("ドキュメントに関する質問をしてください。")
        
        with gr.Row():
            question_input = gr.Textbox(label="質問", placeholder="RAGとは何ですか？", scale=4)
            submit_button = gr.Button("質問する")
            clear_button = gr.Button("クリア")

        with gr.Row():
            answer_output = gr.Textbox(label="回答", lines=5, interactive=False)
            sources_output = gr.Textbox(label="ソース", lines=5, interactive=False)

        submit_button.click(
            fn=answer_question,
            inputs=question_input,
            outputs=[answer_output, sources_output]
        )
        
        clear_button.click(
            fn=lambda: ("", "", ""),
            inputs=None,
            outputs=[question_input, answer_output, sources_output]
        )
        
    print("\n" + "="*60)
    print("💡 Gradio UIを起動します。 http://127.0.0.1:7860 で開いてください。")
    print("="*60)
    demo.launch()

if __name__ == "__main__":
    main()