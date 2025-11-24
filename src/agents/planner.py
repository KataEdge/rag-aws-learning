from typing import List, Dict, Any, Optional
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import BaseTool
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_core.callbacks import BaseCallbackHandler
from pydantic import BaseModel, Field

from ..rag_chain import RAGChain
from ..image_processor import ImageProcessor
from ..aws_utils import BedrockClient, GeminiClient
from ..vector_store import VectorStoreManager

class RAGSearchArgs(BaseModel):
    query: str = Field(description="検索する質問やクエリ")
    k: int = Field(default=3, description="検索する文書数")
    temperature: float = Field(default=0.1, description="生成の温度パラメータ")

class RAGSearchTool(BaseTool):
    args_schema: type = RAGSearchArgs
    name: str = "rag_search"
    description: str = "RAGシステムを使って質問に答える。ドキュメント検索とLLM生成を行う。"

    def __init__(self, rag_chain: RAGChain):
        super().__init__()
        self.rag_chain = rag_chain

    def _run(self, query: str, k: int = 3, temperature: float = 0.1) -> str:
        """RAG検索を実行"""
        try:
            result = self.rag_chain.query(query, k=k, temperature=temperature)
            answer = result.get('answer', '回答が見つかりませんでした')
            sources = result.get('source_documents', [])
            source_text = "\n".join([f"- {doc.metadata.get('source', '不明')}" for doc in sources])
            return f"回答: {answer}\n\nソース: {source_text}"
        except Exception as e:
            return f"RAG検索エラー: {e}"

class ImageAnalysisArgs(BaseModel):
    image_path: str = Field(description="分析する画像ファイルのパス")

class ImageAnalysisTool(BaseTool):
    name: str = "image_analysis"
    description: str = "画像を分析し、テキスト抽出と説明生成を行う。"
    args_schema: type = ImageAnalysisArgs

    def __init__(self, image_processor: ImageProcessor, bedrock_client):
        super().__init__()
        self.image_processor = image_processor
        self.bedrock_client = bedrock_client

    def _run(self, image_path: str) -> str:
        """画像分析を実行"""
        try:
            # OCRテキスト抽出
            ocr_text = self.image_processor.extract_text_from_image(image_path)

            # LLMで説明生成
            description = self.image_processor.get_image_description(image_path, self.bedrock_client)

            return f"OCRテキスト: {ocr_text}\n\n画像説明: {description}"
        except Exception as e:
            return f"画像分析エラー: {e}"

class AWSArgs(BaseModel):
    query: str = Field(description="AWSに関する質問")

class AWSTool(BaseTool):
    name: str = "aws_service"
    description: str = "AWSサービスに関する情報を提供。Bedrockや他のAWSサービスについて説明。"
    args_schema: type = AWSArgs

    def __init__(self, bedrock_client):
        super().__init__()
        self.bedrock_client = bedrock_client

    def _run(self, query: str) -> str:
        """AWS関連の質問に答える"""
        try:
            system_prompt = "あなたはAWSの専門家です。AWSサービスについて正確な情報を提供してください。"
            answer = self.bedrock_client.invoke_claude(query, system_prompt=system_prompt, temperature=0.1)
            return answer
        except Exception as e:
            return f"AWS情報取得エラー: {e}"

class PlannerAgent:
    def __init__(self, rag_chain: RAGChain, image_processor: ImageProcessor, bedrock_client, vector_store: VectorStoreManager):
        self.rag_chain = rag_chain
        self.image_processor = image_processor
        self.bedrock_client = bedrock_client
        self.vector_store = vector_store

        # ツールの初期化
        self.tools = [
            RAGSearchTool(rag_chain),
            ImageAnalysisTool(image_processor, bedrock_client),
            AWSTool(bedrock_client)
        ]

        # エージェントのプロンプト
        self.prompt = PromptTemplate.from_template("""
あなたはPlannerエージェントです。ユーザーのクエリを分析し、実行計画を立てて実行します。

利用可能なツール:
- rag_search: RAGシステムでドキュメント検索と回答生成
- image_analysis: 画像の分析と説明生成
- aws_service: AWSサービスに関する情報提供

クエリを分析し、適切なツールを組み合わせて計画を実行してください。

クエリ: {input}

{agent_scratchpad}
""")

        # ReActエージェントの作成
        self.agent = create_react_agent(
            llm=self._get_llm(),
            tools=self.tools,
            prompt=self.prompt
        )

        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True
        )

    def _get_llm(self):
        """LLMインスタンスを取得"""
        # BedrockClientをLangChain互換にラップ
        from langchain_core.language_models.llms import LLM
        from langchain_core.callbacks.manager import CallbackManagerForLLMRun
        from typing import Iterator

        class BedrockLLM(LLM):
            def __init__(self, bedrock_client):
                super().__init__()
                self.bedrock_client = bedrock_client

            @property
            def _llm_type(self) -> str:
                return "bedrock"

            def _call(self, prompt: str, stop: Optional[List[str]] = None, run_manager: Optional[CallbackManagerForLLMRun] = None) -> str:
                return self.bedrock_client.invoke_claude(prompt, temperature=0.1)

        return BedrockLLM(self.bedrock_client)

    def analyze_and_plan(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        クエリを分析し、実行計画を立てて実行

        Args:
            query: ユーザーのクエリ
            context: 追加のコンテキスト情報

        Returns:
            実行結果
        """
        try:
            # クエリの分析
            analysis_prompt = f"""
以下のクエリを分析してください：

クエリ: {query}

分析項目:
1. クエリの種類（情報検索、画像分析、AWS関連など）
2. 必要なツール
3. 実行順序
4. 期待される出力

分析結果をJSON形式で返してください。
"""

            analysis_result = self.bedrock_client.invoke_claude(analysis_prompt, temperature=0.1)

            # エージェントで実行
            result = self.agent_executor.invoke({"input": query})

            return {
                "query": query,
                "analysis": analysis_result,
                "execution_result": result.get("output", ""),
                "success": True
            }

        except Exception as e:
            return {
                "query": query,
                "error": str(e),
                "success": False
            }

    def get_available_tools(self) -> List[str]:
        """利用可能なツールの一覧を返す"""
        return [tool.name for tool in self.tools]