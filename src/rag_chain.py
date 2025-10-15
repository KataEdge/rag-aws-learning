from typing import Any, Dict, List
from langchain.schema import Document

class RAGChain:
    def __init__(self, vector_store_manager, bedrock_client):
        self.vector_store = vector_store_manager
        self.bedrock = bedrock_client
    
    def create_prompt(self, query: str, context_docs: List[Document]) -> str:
        """
        検索結果を元にプロンプトを作成
        """
        # コンテキストを結合
        context = "\n\n".join([
            f"[文書{i+1}]\n{doc.page_content}" 
            for i, doc in enumerate(context_docs)
        ])
        
        # プロンプトテンプレート
        prompt = f"""以下の文書を参考にして、質問に答えてください。
文書に書かれていない情報については、推測せず「文書には記載されていません」と答えてください。

# 参考文書
{context}

# 質問
{query}

# 回答
"""
        return prompt
    
    def query(self, question: str, k: int = 3) -> Dict[str, Any]:
        """
        RAGパイプライン全体を実行
        
        Args:
            question: ユーザーの質問
            k: 検索する文書数
        
        Returns:
            回答と検索結果を含む辞書
        """
        # 1. ベクトル検索
        print(f"\n🔍 関連文書を検索中... (上位{k}件)")
        relevant_docs = self.vector_store.similarity_search(question, k=k)
        
        print(f"✅ {len(relevant_docs)}件の関連文書を取得")
        
        # 2. プロンプト作成
        prompt = self.create_prompt(question, relevant_docs)
        
        # 3. LLMで回答生成
        print("🤖 LLMで回答を生成中...")
        answer = self.bedrock.invoke_claude(prompt)
        
        # 4. 結果を返す
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