from typing import Any, Dict, List
from langchain.schema import Document

class RAGChain:
    def __init__(self, vector_store_manager, bedrock_client):
        self.vector_store = vector_store_manager
        self.bedrock = bedrock_client

    def rewrite_query(self, query: str, history: List[Dict[str, str]]) -> str:
        """
        会話履歴を考慮して、検索に最適なクエリに書き換える
        """
        if not history:
            return query

        history_text = "\n".join([f"H: {h['human']}\nA: {h['ai']}" for h in history])
        
        system_prompt = "あなたは会話の文脈を理解して、質問をより明確にするアシスタントです。与えられた会話履歴を踏まえて、最後のユーザーの質問を、文脈を補完した自己完結型の質問に書き換えてください。書き換えた質問のみを返してください。"

        user_prompt = f"""# 会話履歴
{history_text}

# 最後の質問
{query}

# 書き換えた質問
"""

        print(f"🔄 クエリ書き換え中... (Temperature: 0.1)")
        rewritten_query = self.bedrock.invoke_claude(user_prompt, system_prompt=system_prompt, temperature=0.1)
        print(f"🔄 クエリを書き換え: '{query}' -> '{rewritten_query}'")
        return rewritten_query

    def create_prompt(self, query: str, context_docs: List[Document], history: List[Dict[str, str]]) -> tuple[str, str]:
        """
        検索結果と会話履歴を元にプロンプトを作成
        Returns: (system_prompt, user_prompt)
        """
        # システムプロンプト
        system_prompt = """あなたはRAG（Retrieval-Augmented Generation）システムのアシスタントです。
以下の参考文書と会話履歴を踏まえて、ユーザーの質問に答えてください。

回答の原則:
- 参考文書と会話履歴から関連情報を優先的に使用してください
- 文書に記載されていない情報については、一般知識を使用しても構いません
- 情報源が不明確な場合は「文書には記載されていません」と明確に述べてください
- 回答は正確で、根拠に基づいたものにしてください"""

        # コンテキストを結合
        context = "\n\n".join([
            f"[文書{i+1}]\n{doc.page_content}"
            for i, doc in enumerate(context_docs)
        ])

        # 会話履歴を整形
        history_text = "\n".join([f"Human: {h['human']}\nAssistant: {h['ai']}" for h in history])

        # ユーザープロンプト
        user_prompt = f"""# 参考文書
{context}

# 会話履歴
{history_text}

# 質問
{query}"""

        return system_prompt, user_prompt

    def query(self, question: str, history: List[Dict[str, str]] = [], k: int = 3, temperature: float = 0.1, filter: Dict[str, Any] = None, images: List[bytes] = None) -> Dict[str, Any]:
        """
        RAGパイプライン全体を実行（マルチモーダル対応）

        Args:
            question: ユーザーの質問
            history: 会話履歴
            k: 検索する文書数
            filter: メタデータフィルタ
            images: 質問に付随する画像データ（バイト列のリスト）

        Returns:
            回答と検索結果を含む辞書
        """
        # 1. クエリ書き換え
        rewritten_question = self.rewrite_query(question, history)

        # 2. ハイブリッド検索
        print(f"\n🔍 関連文書をハイブリッド検索中... (上位{k}件)")
        hybrid_retriever = self.vector_store.get_hybrid_retriever()
        if hybrid_retriever:
            all_docs = hybrid_retriever.invoke(rewritten_question)
            relevant_docs = all_docs[:k]
        else:
            # フォールバック: ベクトル検索
            print("ハイブリッド検索が利用できないため、ベクトル検索にフォールバックします")
            relevant_docs = self.vector_store.similarity_search(rewritten_question, k=k, filter=filter)

        print(f"✅ {len(relevant_docs)}件の関連文書を取得")

        # 3. プロンプト作成
        system_prompt, user_prompt = self.create_prompt(question, relevant_docs, history)

        # 4. LLMで回答生成
        print(f"🤖 LLMで回答を生成中... (Temperature: {temperature})")
        answer = self.bedrock.invoke_claude(user_prompt, system_prompt=system_prompt, temperature=temperature, images=images)

        # 5. 結果を返す
        return {
            'question': question,
            'answer': answer,
            'source_documents': relevant_docs,
            'prompt': user_prompt  # デバッグ用
        }
    
    def pretty_print_result(self, result: Dict[str, Any]):
        """
        結果を見やすく表示
        """
        print("\n" + "="*60)
        print(f"💬 質問: {result['question']}")
        print("="*60)
        
        print(f"\n📝 回答:\n{result['answer']}")
        
        print(f"\n📚 参考文書 ({len(result['source_documents'])}件):")
        for i, doc in enumerate(result['source_documents'], 1):
            print(f"\n--- 文書{i} ---")
            print(doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content)
        
        print("\n" + "="*60)