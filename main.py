import streamlit as st
from src.document_loader import DocumentLoader
from src.embeddings import EmbeddingManager
from src.vector_store import VectorStoreManager
from src.aws_utils import BedrockClient
from src.rag_chain import RAGChain
from src.s3_manager import S3Manager
from dotenv import load_dotenv
import os
import shutil

# ---
# Streamlitのキャッシュ機能を使ってRAGチェーンを効率的に初期化
# ---
@st.cache_resource
def initialize_rag_system():
    """RAGシステムを初期化"""
    
    # 環境変数読み込み
    load_dotenv()
    
    print("="*60)
    print("🚀 RAG + AWS Bedrockシステムの起動")
    print("="*60)
    
    # ドキュメントディレクトリを準備
    os.makedirs('documents', exist_ok=True)

    # 1. 埋め込みモデル準備
    print("\n🤖 ステップ1: 埋め込みモデル準備")
    embedding_manager = EmbeddingManager()
    embeddings = embedding_manager.get_embeddings()
    
    # 2. ベクトルストア準備
    print("\n💾 ステップ2: ベクトルストア準備")
    vector_manager = VectorStoreManager(embeddings)
    
    # 永続化されたベクトルストアがあれば読み込み、なければ新規作成
    try:
        vector_manager.load_vectorstore()
        print("✅ 既存のベクトルストアを読み込みました")
    except FileNotFoundError:
        print("🤔 既存のベクトルストアが見つからないため、新規に作成します")
        # ドキュメント読み込み
        print("📄 ドキュメント処理中...")
        loader = DocumentLoader()
        documents = loader.load_directory('documents')
        print(f"✅ {len(documents)}個のチャンクを作成")
        
        vector_manager.create_vectorstore(documents)
        print("✅ 新規ベクトルストアの作成完了")
    
    # 4. Bedrockクライアント準備
    print("\n🔌 ステップ4: AWS Bedrock接続")
    bedrock_client = BedrockClient(region_name=os.getenv('AWS_REGION', 'ap-northeast-1'))
    
    # 5. RAGチェーン作成
    print("\n⛓️  ステップ5: RAGチェーン構築")
    rag_chain_instance = RAGChain(vector_manager, bedrock_client)
    print("✅ RAGシステム準備完了")
    
    return rag_chain_instance, vector_manager

def answer_question(rag_chain, question, history):
    """質問に回答する"""
    if rag_chain is None:
        return "エラー: RAGシステムが初期化されていません。"

    print(f"\n🤔 質問受信: {question}")
    
    # LangChainの形式に履歴を変換
    langchain_history = [{"human": h, "ai": a} for h, a in history]
    
    result = rag_chain.query(question, history=langchain_history, k=2)
    
    answer = result.get('answer', "回答が見つかりませんでした。")
    
    # ソース情報を整形
    source_documents = result.get('source_documents', [])
    sources_text = "\n\n---\n\n**ソース:**\n"
    if source_documents:
        for doc in source_documents:
            source_info = doc.metadata.get('source', '不明なソース')
            sources_text += f"- {source_info}\n"
    else:
        sources_text += "関連するソースは見つかりませんでした。"
    
    full_answer = f"{answer}{sources_text}"
    
    print("✅ 回答生成完了")
    return full_answer

def main():
    st.set_page_config(page_title="AWS Bedrock RAG Demo", page_icon="🤖")
    st.title("🤖 AWS Bedrock RAG デモアプリ")
    st.markdown("ドキュメントに関する質問をしてください。")

    # RAGシステムの初期化
    try:
        rag_chain, vector_manager = initialize_rag_system()
        st.success("RAGシステムの準備が完了しました。")
    except Exception as e:
        st.error(f"RAGシステムの初期化中にエラーが発生しました: {e}")
        st.stop()

    # --- サイドバーのUI ---
    with st.sidebar:
        st.header("ナレッジ管理")
        st.markdown("ファイルを追加して、RAGシステムの知識を更新します。")
        
        uploaded_files = st.file_uploader(
            "文書ファイルを選択",
            accept_multiple_files=True,
            type=['pdf', 'txt', 'docx', 'xlsx'],
            help="PDF, テキスト, Word, Excelファイルをアップロードできます。"
        )

        if st.button("アップロードして知識を更新", type="primary"):
            if uploaded_files:
                temp_dir = "temp_uploaded_files"
                
                with st.spinner(f"{len(uploaded_files)}個のファイルを処理中..."):
                    try:
                        # 一時ディレクトリを作成
                        if not os.path.exists(temp_dir):
                            os.makedirs(temp_dir)

                        # ファイルを一時保存
                        for uploaded_file in uploaded_files:
                            file_path = os.path.join(temp_dir, uploaded_file.name)
                            with open(file_path, "wb") as f:
                                f.write(uploaded_file.getvalue())

                        # ドキュメントを読み込んでチャンクに分割
                        loader = DocumentLoader()
                        documents = loader.load_directory(temp_dir)

                        if documents:
                            # ベクトルストアに追加
                            vector_manager.add_documents(documents)
                            st.success(f"知識ベースを更新しました ({len(documents)}チャンク追加)")
                        else:
                            st.warning("アップロードされたファイルからテキストを抽出できませんでした。")

                    except Exception as e:
                        st.error(f"処理中にエラーが発生しました: {e}")
                    
                    finally:
                        # 一時ディレクトリをクリーンアップ
                        if os.path.exists(temp_dir):
                            shutil.rmtree(temp_dir)

            else:
                st.warning("ファイルが選択されていません。")

    # チャット履歴をセッション状態で管理
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 過去のメッセージを表示
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # ユーザーからの入力を受け取る
    if prompt := st.chat_input("RAGとは何ですか？"):
        # ユーザーのメッセージを履歴に追加して表示
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # AIの応答を生成
        with st.chat_message("assistant"):
            with st.spinner("回答を生成中です..."):
                # Streamlitの履歴形式からLangChainの履歴形式へ変換
                history_for_chain = [(msg["content"]) for msg in st.session_state.messages if msg["role"] == "user"]
                history_pairs = []
                if len(history_for_chain) > 1:
                     # 最後の質問は除く
                    history_pairs = list(zip(history_for_chain[:-1:2], history_for_chain[1::2]))


                response = answer_question(rag_chain, prompt, history_pairs)
                st.markdown(response)
        
        # AIの応答を履歴に追加
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()