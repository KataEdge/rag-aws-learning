
import streamlit as st
from src.document_loader import DocumentLoader
from src.embeddings import EmbeddingManager
from src.vector_store import VectorStoreManager
from src.aws_utils import BedrockClient, GeminiClient
from src.rag_chain import RAGChain
from src.s3_manager import S3Manager
from src.image_processor import ImageProcessor
from dotenv import load_dotenv
import os
import shutil
import json
from datetime import datetime

# ---
# Streamlitのキャッシュ機能を使ってRAGチェーンを効率的に初期化
# ---
@st.cache_resource
def initialize_rag_system(model_id):
    """RAGシステムを初期化"""

    # 環境変数読み込み
    load_dotenv()

    print("="*60)
    print("🚀 RAG + AWS Bedrockシステムの起動")
    print("="*60)

    # ドキュメントディレクトリを準備
    documents_dir = os.path.join(os.path.dirname(__file__), 'documents')
    os.makedirs(documents_dir, exist_ok=True)

    # 1. 埋め込みモデル準備
    print("\n🤖 ステップ1: 埋め込みモデル準備")
    embedding_manager = EmbeddingManager()
    embeddings = embedding_manager.get_embeddings()

    # 2. ベクトルストア準備
    print("\n💾 ステップ2: ベクトルストア準備")
    vector_manager = VectorStoreManager(embeddings)

    # モデルタイプに応じてクライアントを作成
    if 'gemini' in model_id:
        client = GeminiClient(model_name=model_id)
    else:
        client = BedrockClient(region_name=os.getenv('AWS_REGION', 'ap-northeast-1'), model_id=model_id)

    # 永続化されたベクトルストアがあれば読み込み、なければ新規作成
    try:
        vector_manager.load_vectorstore()
        print("✅ 既存のベクトルストアを読み込みました")
    except FileNotFoundError:
        print("🤔 既存のベクトルストアが見つからないため、新規に作成します")
        # ドキュメント読み込み
        print("📄 ドキュメント処理中...")
        loader = DocumentLoader(bedrock_client=client)
        documents = loader.load_directory(documents_dir)
        print(f"✅ {len(documents)}個のチャンクを作成")

        vector_manager.create_vectorstore(documents)
        print("✅ 新規ベクトルストアの作成完了")


    # モデル利用可能性チェック
    print(f"🔍 モデル {model_id} の利用可能性を確認中...")
    try:
        client.test_connection()
        print("✅ モデル利用可能")
    except Exception as e:
        error_msg = f"選択されたモデル '{model_id}' は利用できません: {e}"
        print(f"❌ {error_msg}")
        raise ValueError(error_msg)

    # 5. RAGチェーン作成
    print("\n⛓️  ステップ5: RAGチェーン構築")
    rag_chain_instance = RAGChain(vector_manager, client)
    print("✅ RAGシステム準備完了")

    # 6. 画像処理とエージェント初期化
    # print("\n🤖 ステップ6: エージェントシステム初期化")
    # print("✅ RAGシステム準備完了")

    return rag_chain_instance, vector_manager

def save_feedbacks():
    """フィードバックをJSONファイルに保存"""
    with open("feedbacks.json", "w", encoding="utf-8") as f:
        json.dump(st.session_state.feedbacks, f, ensure_ascii=False, indent=2)

def answer_question(rag_chain, question, history, temperature=0.1, k=2):
    """質問に回答する"""
    if rag_chain is None:
        return "エラー: RAGシステムが初期化されていません。"

    print(f"\n🤔 質問受信: {question}")

    # LangChainの形式に履歴を変換
    langchain_history = [{"human": h, "ai": a} for h, a in history]

    result = rag_chain.query(question, history=langchain_history, k=k, temperature=temperature)

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
    st.title("🤖 RAG デモアプリ")
    st.markdown("ドキュメントに関する質問をしてください。")

    # フィードバックの初期化
    if "feedbacks" not in st.session_state:
        st.session_state.feedbacks = []
        # 既存のフィードバックを読み込み
        if os.path.exists("feedbacks.json"):
            try:
                with open("feedbacks.json", "r", encoding="utf-8") as f:
                    st.session_state.feedbacks = json.load(f)
            except:
                pass

    if "feedback_counter" not in st.session_state:
        st.session_state.feedback_counter = 0

    # デフォルト値の設定
    selected_model_id = os.getenv('BEDROCK_MODEL_ID', 'gemini-2.5-flash')
    temperature = 0.1
    k = 2
    file_type_filter = "すべて"
    folder_filter = ""
    use_agent = False  # エージェント使用フラグ

    # モデル選択の準備
    available_models = {
        "Claude 3 Haiku": "anthropic.claude-3-haiku-20240307-v1:0",
        "Amazon Nova Lite": "amazon.nova-lite-v1:0",
        "gemini-2.5-flash": "gemini-2.5-flash",
    }

    # デフォルトモデル名を取得
    default_model_id = os.getenv('BEDROCK_MODEL_ID', 'gemini-2.5-flash')
    default_model_name = None
    for name, model_id in available_models.items():
        if model_id == default_model_id:
            default_model_name = name
            break
    if default_model_name is None:
        default_model_name = "gemini-2.5-flash"  # フォールバック

    # RAGシステムの初期化
    try:
        rag_chain, vector_manager = initialize_rag_system(model_id=selected_model_id)
        st.success(f"RAGシステムの準備が完了しました。使用モデル: {default_model_name}")
    except Exception as e:
        st.error(f"RAGシステムの初期化中にエラーが発生しました: {e}")
        st.stop()

    # --- サイドバーのUI ---
    with st.sidebar:
        # タブでサイドバーを整理
        tab1, tab2, tab3 = st.tabs(["🤖 モデル設定", "🔍 検索設定", "📚 ナレッジ管理"])

        with tab1:
            st.header("🤖 モデル設定")

            selected_model_name = st.selectbox(
                "LLMモデル",
                options=list(available_models.keys()),
                index=list(available_models.keys()).index(default_model_name),
                help="使用するLLMモデルを選択してください。"
            )

            selected_model_id = available_models[selected_model_name]

            # Temperature設定
            temperature = st.slider(
                "Temperature (創造性)",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.1,
                help="低い値（0.0-0.3）：事実に基づいた正確な回答、高い値（0.7-1.0）：創造的な回答"
            )

            # エージェント使用設定

        with tab2:
            st.header("🔍 検索設定")

            # 検索文書数設定
            k = st.slider(
                "検索文書数 (k)",
                min_value=1,
                max_value=10,
                value=2,
                step=1,
                help="検索して使用する関連文書の数を設定します。多いほど多様な情報が考慮されますが、処理時間が長くなります。"
            )

            # メタデータフィルタ設定
            st.subheader("検索フィルタ")
            file_type_filter = st.selectbox(
                "ファイルタイプ",
                options=["すべて", ".pdf", ".txt", ".docx", ".xlsx"],
                index=0,
                help="検索対象を特定のファイルタイプに絞り込みます。"
            )
            folder_filter = st.text_input(
                "フォルダ名 (完全一致)",
                placeholder="例: /Users/username/Downloads/sales",
                help="検索対象を特定のフォルダ内のファイルに絞り込みます。完全パスを入力してください。"
            )

            # 画像アップロードUI
            st.subheader("📸 画像アップロード（オプション）")
            uploaded_image_files = st.file_uploader(
                "質問に関連する画像をアップロードしてください",
                type=['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp'],
                accept_multiple_files=True,
                help="画像をアップロードすると、LLMが画像の内容を理解して回答します。"
            )

            if st.button("画像をクリア", type="secondary"):
                st.session_state.uploaded_images = []
                st.rerun()

            # アップロードされた画像を表示
            if uploaded_image_files:
                st.session_state.uploaded_images = []
                for uploaded_file in uploaded_image_files:
                    image_bytes = uploaded_file.getvalue()
                    st.session_state.uploaded_images.append(image_bytes)
                    st.image(image_bytes, caption=uploaded_file.name, width=150)

        with tab3:
            st.header("📚 ナレッジ管理")
            st.markdown("ファイルを追加して、RAGシステムの知識を更新します。")

            # データベースクリアボタン
            if st.button("🗑️ データベースをクリア", type="secondary", help="Chromaデータベースを完全に削除します。"):
                try:
                    vector_manager.clear_database()
                    st.success("データベースをクリアしました。アプリを再起動してください。")
                    st.rerun()
                except Exception as e:
                    st.error(f"データベースクリア中にエラーが発生しました: {e}")

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
                                # ベクトルストアに同期（差分更新）
                                result = vector_manager.sync_documents(documents)
                                st.info(f"追加: {result['num_added']}, 更新: {result['num_updated']}, スキップ: {result['num_skipped']}, 削除: {result['num_deleted']}")
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

    # 画像アップロード用の状態管理
    if "uploaded_images" not in st.session_state:
        st.session_state.uploaded_images = []

    # 過去のメッセージを表示
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # ユーザーからの入力を受け取る
    if prompt := st.chat_input("RAGとは何ですか？ または画像をアップロードして質問してください。"):
        # ユーザーのメッセージを履歴に追加して表示
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # AIの応答を生成
        with st.chat_message("assistant"):
            with st.spinner("回答を生成中です..."):
                # 会話履歴を (human, ai) のペアに変換（最後のユーザーメッセージを除く）
                history_pairs = []
                messages = st.session_state.messages[:-1]  # 最後のユーザーメッセージを除く
                for i in range(0, len(messages) - 1, 2):
                    if i + 1 < len(messages) and messages[i]["role"] == "user" and messages[i + 1]["role"] == "assistant":
                        history_pairs.append((messages[i]["content"], messages[i + 1]["content"]))


                # フィルタ構築
                filter_dict = {}
                if file_type_filter != "すべて":
                    filter_dict["file_type"] = file_type_filter
                if folder_filter:
                    filter_dict["folder"] = folder_filter

                filter_param = filter_dict if filter_dict else None

                # アップロードされた画像を渡す
                images_param = st.session_state.uploaded_images if st.session_state.uploaded_images else None

                response = answer_question(rag_chain, prompt, history_pairs, temperature, k)

                st.markdown(response)

                # フィードバックボタン
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("👍 Good", key=f"good_{st.session_state.feedback_counter}"):
                        feedback = {
                            "question": prompt,
                            "answer": response,
                            "feedback": "good",
                            "timestamp": datetime.now().isoformat()
                        }
                        st.session_state.feedbacks.append(feedback)
                        save_feedbacks()
                        st.success("フィードバックを記録しました！")

                with col2:
                    if st.button("👎 Bad", key=f"bad_{st.session_state.feedback_counter}"):
                        feedback = {
                            "question": prompt,
                            "answer": response,
                            "feedback": "bad",
                            "timestamp": datetime.now().isoformat()
                        }
                        st.session_state.feedbacks.append(feedback)
                        save_feedbacks()
                        st.success("フィードバックを記録しました！")

                st.session_state.feedback_counter += 1

        # AIの応答を履歴に追加
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()