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
        
        prompt = f"""H: 以下の会話履歴を踏まえて、最後のユーザーの質問を、文脈を補完した自己完結型の質問に書き換えてください。
書き換えた質問のみを返してください。

# 会話履歴
{history_text}

# 最後の質問
{query}

# 書き換えた質問
"""
        
        rewritten_query = self.bedrock.invoke_claude(prompt)
        print(f"🔄 クエリを書き換え: '{query}' -> '{rewritten_query}'")
        return rewritten_query

    def create_prompt(self, query: str, context_docs: List[Document], history: List[Dict[str, str]]) -> str:
        """
        検索結果と会話履歴を元にプロンプトを作成
        """
        # コンテキストを結合
        context = "\n\n".join([
            f"[文書{i+1}]\n{doc.page_content}"
            for i, doc in enumerate(context_docs)
        ])
        
        # 会話履歴を整形
        history_text = "\n".join([f"H: {h['human']}\nA: {h['ai']}" for h in history])

        # プロンプトテンプレート
        prompt = f"""H: 以下の参考文書と会話履歴を踏まえて、最後の質問に答えてください。
回答はまず参考文書と会話履歴から生成してください。
もし参考文書や会話履歴に該当する情報がない場合は、あなたが持つ一般的な知識を用いて回答しても構いません。
ただし、文書に書かれていない情報については、推測せず「文書には記載されていません」と明確に述べてください。

# 参考文書
{context}

# 会話履歴
{history_text}

# 質問
{query}

A: """
        return prompt

    def query(self, question: str, history: List[Dict[str, str]] = [], k: int = 3) -> Dict[str, Any]:
        """
        RAGパイプライン全体を実行
        
        Args:
            question: ユーザーの質問
            history: 会話履歴
            k: 検索する文書数
        
        Returns:
            回答と検索結果を含む辞書
        """
        # 1. クエリ書き換え
        rewritten_question = self.rewrite_query(question, history)

        # 2. ベクトル検索
        print(f"\n🔍 関連文書を検索中... (上位{k}件)")
        relevant_docs = self.vector_store.similarity_search(rewritten_question, k=k)
        
        print(f"✅ {len(relevant_docs)}件の関連文書を取得")
        
        # 3. プロンプト作成
        prompt = self.create_prompt(question, relevant_docs, history) # 元の質問をプロンプトに使う
        
        # 4. LLMで回答生成
        print("🤖 LLMで回答を生成中...")
        answer = self.bedrock.invoke_claude(prompt)
        
        # 5. 結果を返す
        return {
            'question': question,
            'answer': answer,
            'source_documents': relevant_docs,
            'prompt': prompt  # デバッグ用
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