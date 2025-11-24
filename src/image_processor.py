import os
import logging
from typing import List, Dict, Any, Optional
from PIL import Image
import pytesseract
import cv2
import numpy as np
from langchain.schema import Document

# ログ設定
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class ImageProcessor:
    def __init__(self):
        # Tesseractの設定（必要に応じてパスを設定）
        # pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'  # Linuxの場合
        pass

    def extract_text_from_image(self, image_path: str) -> str:
        """
        画像からテキストを抽出（OCR）
        """
        try:
            # PILで画像を開く
            image = Image.open(image_path)

            # 画像を前処理
            processed_image = self._preprocess_image_for_ocr(image)

            # OCR実行
            text = pytesseract.image_to_string(processed_image, lang='jpn+eng')

            logging.info(f"画像からテキストを抽出: {image_path} ({len(text)}文字)")
            return text.strip()

        except Exception as e:
            logging.error(f"画像からのテキスト抽出に失敗: {image_path} - {e}")
            return ""

    def _preprocess_image_for_ocr(self, image: Image.Image) -> Image.Image:
        """
        OCRのための画像前処理
        """
        # OpenCVに変換
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # グレースケール変換
        gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)

        # ノイズ除去
        gray = cv2.medianBlur(gray, 3)

        # 二値化（適応的閾値処理）
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

        # PIL Imageに戻す
        pil_image = Image.fromarray(thresh)
        return pil_image

    def get_image_description(self, image_path: str, bedrock_client=None) -> str:
        """
        LLMを使って画像の説明を生成
        """
        if not bedrock_client:
            return "LLMクライアントが利用できません"

        try:
            # 画像を読み込み
            with open(image_path, 'rb') as f:
                image_bytes = f.read()

            prompt = """この画像について詳細に説明してください。
- 画像の内容を詳しく記述
- 重要な要素や特徴を指摘
- 画像の種類（写真、イラスト、図表など）を特定
- テキストが含まれる場合はその内容も記述"""

            description = bedrock_client.invoke_claude(
                prompt,
                images=[image_bytes],
                max_tokens=500,
                temperature=0.3
            )

            logging.info(f"画像説明を生成: {image_path}")
            return description.strip()

        except Exception as e:
            logging.error(f"画像説明生成に失敗: {image_path} - {e}")
            return f"画像説明の生成に失敗しました: {e}"

    def create_image_document(self, image_path: str, bedrock_client=None) -> Document:
        """
        画像からDocumentオブジェクトを作成
        """
        # OCRでテキスト抽出
        ocr_text = self.extract_text_from_image(image_path)

        # LLMで説明生成
        description = self.get_image_description(image_path, bedrock_client)

        # コンテンツの結合
        content_parts = []
        if description:
            content_parts.append(f"画像説明: {description}")
        if ocr_text:
            content_parts.append(f"OCRテキスト: {ocr_text}")

        content = "\n\n".join(content_parts) if content_parts else "画像からテキストを抽出できませんでした"

        # メタデータ
        metadata = {
            'source': image_path,
            'file_name': os.path.basename(image_path),
            'file_type': os.path.splitext(image_path)[1].lower(),
            'folder': os.path.dirname(image_path),
            'content_type': 'image',
            'has_ocr_text': bool(ocr_text),
            'has_description': bool(description)
        }

        return Document(
            page_content=content,
            metadata=metadata
        )

    def process_image_directory(self, directory_path: str, bedrock_client=None) -> List[Document]:
        """
        ディレクトリ内の画像ファイルを処理
        """
        documents = []
        supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)

            # 画像ファイルのみ処理
            if os.path.isfile(file_path) and os.path.splitext(filename)[1].lower() in supported_extensions:
                try:
                    doc = self.create_image_document(file_path, bedrock_client)
                    documents.append(doc)
                    logging.info(f"画像ファイルを処理: {filename}")
                except Exception as e:
                    logging.error(f"画像ファイルの処理に失敗: {filename} - {e}")

        logging.info(f"画像ディレクトリ処理完了: {directory_path} ({len(documents)}ファイル)")
        return documents

    def get_image_embedding(self, image_path: str) -> Optional[np.ndarray]:
        """
        画像の特徴ベクトルを抽出（将来の拡張用）
        """
        # ここでは簡易的な実装
        # 実際にはCLIPや他の画像埋め込みモデルを使用
        try:
            image = Image.open(image_path)
            image = image.resize((224, 224))  # 標準サイズにリサイズ
            image_array = np.array(image)

            # 簡易的な特徴抽出（平均色など）
            features = image_array.mean(axis=(0, 1))  # RGBの平均

            return features

        except Exception as e:
            logging.error(f"画像特徴抽出に失敗: {image_path} - {e}")
            return None