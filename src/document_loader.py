import logging
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from unstructured.partition.auto import partition
from unstructured.chunking.title import chunk_by_title
import os
import gc
import psutil
from typing import List, Dict, Any, Iterator, Optional
from langchain.schema import Document

# ログ設定
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class DocumentLoader:
    def __init__(self, chunk_size=1000, chunk_overlap=200, bedrock_client=None, batch_size=30, memory_threshold_mb=1500):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        self.bedrock_client = bedrock_client
        self.batch_size = batch_size
        self.memory_threshold_mb = memory_threshold_mb
    
    def load_pdf(self, file_path):
        """PDFファイルを読み込み（最適化版）"""
        try:
            # unstructuredでパーティション分割（PDF対応）
            elements = partition(filename=file_path)

            # 表要素の特別処理
            elements = self._process_table_elements(elements, file_path)

            # chunk_by_title で構造を保持したチャンキング
            chunks = chunk_by_title(
                elements,
                combine_text_under_n_chars=500,
                max_characters=2000,
                new_after_n_chars=1500,
            )

            # LangChain Document に変換
            documents = []
            for chunk in chunks:
                metadata = self._extract_metadata(chunk, file_path)
                enriched_text = self._enrich_text_with_metadata(chunk.text, metadata)

                doc = Document(
                    page_content=enriched_text,
                    metadata=metadata
                )
                documents.append(doc)

            logging.info(f"PDFファイルを読み込み: {file_path} ({len(documents)}チャンク)")
            return documents

        except Exception as e:
            logging.error(f"PDFファイルの読み込み中にエラー: {file_path} - {e}")
            # フォールバック
            try:
                loader = PyPDFLoader(file_path)
                documents = loader.load()
                return self.text_splitter.split_documents(documents)
            except:
                return []

    def load_text(self, file_path):
        """テキストファイルを読み込み（最適化版）"""
        try:
            # テキストファイルもunstructuredで処理
            elements = partition(filename=file_path)

            # chunk_by_title で構造を保持したチャンキング（テキストにも適用）
            chunks = chunk_by_title(
                elements,
                combine_text_under_n_chars=500,
                max_characters=2000,
                new_after_n_chars=1500,
            )

            # LangChain Document に変換
            documents = []
            for chunk in chunks:
                metadata = self._extract_metadata(chunk, file_path)
                enriched_text = self._enrich_text_with_metadata(chunk.text, metadata)

                doc = Document(
                    page_content=enriched_text,
                    metadata=metadata
                )
                documents.append(doc)

            logging.info(f"テキストファイルを読み込み: {file_path} ({len(documents)}チャンク)")
            return documents

        except Exception as e:
            logging.error(f"テキストファイルの読み込み中にエラー: {file_path} - {e}")
            # フォールバック
            try:
                loader = TextLoader(file_path, encoding='utf-8')
                documents = loader.load()
                return self.text_splitter.split_documents(documents)
            except:
                return []

    def load_unstructured_file(self, file_path):
        """WordやExcelなどの非構造化ファイルを読み込み"""
        try:
            loader = UnstructuredFileLoader(file_path)
            documents = loader.load()
            logging.info(f"非構造化ファイルを読み込み: {file_path}")
            return self.text_splitter.split_documents(documents)
        except Exception as e:
            logging.error(f"非構造化ファイルの読み込み中にエラー: {file_path} - {e}")
            return []

    def _process_table_elements(self, elements, file_path):
        """表要素を特別処理"""
        processed_elements = []
        for element in elements:
            if hasattr(element, 'category') and element.category == 'Table':
                # 表をHTML形式に変換
                html_content = self._table_to_html(element)
                element.text = html_content

                # LLMで要約（オプション）
                if self.bedrock_client:
                    summary = self._summarize_table_with_llm(element, file_path)
                    if summary:
                        element.text = f"表の要約: {summary}\n\n元の表:\n{html_content}"

            processed_elements.append(element)
        return processed_elements

    def _table_to_html(self, table_element):
        """表要素をHTMLに変換"""
        try:
            # unstructuredの表要素をHTMLに変換
            html_table = table_element.metadata.text_as_html if hasattr(table_element, 'metadata') and table_element.metadata.text_as_html else str(table_element)
            return f"<table>\n{html_table}\n</table>"
        except:
            return str(table_element)

    def _summarize_table_with_llm(self, table_element, file_path):
        """LLMで表を要約"""
        try:
            system_prompt = "あなたは表の内容を分析して、自然言語で簡潔に要約するアシスタントです。"

            user_prompt = f"""以下の表の内容を、自然言語で簡潔に要約してください。
ファイル: {os.path.basename(file_path)}

表の内容:
{table_element.text}

要約:"""

            summary = self.bedrock_client.invoke_claude(user_prompt, system_prompt=system_prompt, max_tokens=300)
            return summary.strip()
        except Exception as e:
            logging.warning(f"表の要約に失敗: {e}")
            return None

    def _extract_metadata(self, chunk, file_path):
        """チャンクからメタデータを抽出"""
        metadata = {
            'source': file_path,
            'file_name': os.path.basename(file_path),
        }

        # チャンクのメタデータから親ヘッダー情報を抽出
        if hasattr(chunk, 'metadata') and chunk.metadata:
            if hasattr(chunk.metadata, 'parent_id'):
                metadata['parent_id'] = chunk.metadata.parent_id
            if hasattr(chunk.metadata, 'category_depth'):
                metadata['category_depth'] = chunk.metadata.category_depth

        # テキストから見出し情報を抽出（簡易版）
        text = chunk.text[:200]  # 先頭200文字で判定
        if '章:' in text or '節:' in text or '項:' in text:
            # 日本語の見出しパターンを検出
            lines = text.split('\n')
            for line in lines[:3]:  # 最初の3行をチェック
                if any(keyword in line for keyword in ['章', '節', '項', '第', '条']):
                    metadata['section_header'] = line.strip()
                    break

        return metadata

    def _enrich_text_with_metadata(self, text, metadata):
        """テキストにメタデータを注入"""
        enriched_parts = []

        # ファイル名を追加
        if 'file_name' in metadata:
            enriched_parts.append(f"ファイル: {metadata['file_name']}")

        # セクションヘッダーを追加
        if 'section_header' in metadata:
            enriched_parts.append(f"セクション: {metadata['section_header']}")

        # 親IDがある場合は追加
        if 'parent_id' in metadata:
            enriched_parts.append(f"親要素ID: {metadata['parent_id']}")

        # メタデータをテキストの先頭に追加
        if enriched_parts:
            metadata_prefix = " > ".join(enriched_parts) + "\n\n"
            return metadata_prefix + text

        return text

    def load_directory(self, directory_path):
        """ディレクトリ内の全ファイルを読み込み（拡張版）"""
        all_documents = []
        supported_count = 0
        skipped_count = 0

        # python-magicのインポートチェック
        try:
            import magic
            HAS_MAGIC = True
        except ImportError:
            HAS_MAGIC = False

        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)

            # 隠しファイルやディレクトリはスキップ
            if filename.startswith('.') or os.path.isdir(file_path):
                continue

            try:
                if filename.endswith('.pdf'):
                    all_documents.extend(self.load_pdf(file_path))
                    supported_count += 1
                elif filename.endswith(('.txt', '.md', '.markdown', '.py', '.js', '.ts', '.html', '.htm', '.css', '.json', '.xml', '.yaml', '.yml', '.csv')):
                    all_documents.extend(self.load_text(file_path))
                    supported_count += 1
                elif filename.endswith(('.docx', '.xlsx', '.doc', '.xls', '.rtf', '.odt', '.ods')):
                    all_documents.extend(self.load_unstructured_file(file_path))
                    supported_count += 1
                else:
                    # python-magicが利用可能な場合はMIMEタイプで判定
                    if HAS_MAGIC:
                        try:
                            mime_type = magic.from_file(file_path, mime=True)
                            if mime_type and (
                                mime_type.startswith('text/') or
                                mime_type in ['application/json', 'application/xml', 'application/rtf', 'application/vnd.openxmlformats-officedocument']
                            ):
                                all_documents.extend(self.load_text(file_path))
                                supported_count += 1
                            else:
                                skipped_count += 1
                                logging.debug(f"サポート外のMIMEタイプ: {file_path} ({mime_type})")
                        except Exception as e:
                            skipped_count += 1
                            logging.debug(f"MIMEタイプ検出失敗: {file_path} - {e}")
                    else:
                        skipped_count += 1
                        logging.debug(f"サポートされていないファイル形式: {file_path}")

            except Exception as e:
                logging.error(f"ファイルの処理中にエラー: {file_path} - {e}")
                skipped_count += 1

        logging.info(f"ディレクトリ内のファイルを読み込み完了: {directory_path}")
        logging.info(f"処理ファイル: {supported_count}個, スキップ: {skipped_count}個")
        return all_documents