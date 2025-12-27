# RAG AWS Learning - 多様なLLMを活用したマルチモーダルRAGデモアプリケーション

このプロジェクトは、Retrieval-Augmented Generation (RAG) を使用したドキュメント検索・質問応答システムのデモアプリケーションです。AWS Bedrock、Google GeminiなどのLLMをサポートし、マルチモーダル（テキスト+画像）処理に対応しています。

## 特徴

- **多様なLLMサポート**: AWS Bedrock (Claude, Nova)、Google Gemini 2.5 Flash
- **マルチモーダル対応**: テキストドキュメントと画像の同時処理
- **高度なチャンキング**: 構造保持型チャンキング（Unstructured + ルールベース）
- **ハイブリッド検索**: BM25キーワード検索 + ベクトル検索 + メタデータフィルタ
- **Streamlit UI**: タブ分けされた直感的なWebインターフェース
- **会話履歴**: コンテキストを保持した継続的な対話
- **動的パラメータ調整**: Temperature、検索文書数、フィルタのリアルタイム変更
- **ドキュメント管理**: 差分更新による効率的な知識ベース管理、データベースのクリア機能
- **フィードバックシステム**: 回答に対するGood/Bad評価機能

## アーキテクチャ

```text
Documents/Images → Document Loader → Chunking → Embeddings → Vector Store → Hybrid Retriever → LLM → Answer
     ↓
  Image Processor → OCR/Description → Document
```

## 必要条件

- Python 3.8+
- AWSアカウント（Bedrock利用時）
- Google AI APIキー（Gemini利用時）
- Tesseract OCR（画像処理時）
- libmagic（ファイルタイプ検出用）

## インストール

1. リポジトリをクローンまたはダウンロード

2. 仮想環境を作成・有効化

   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. システム依存関係をインストール

   **Tesseract OCR (画像処理用)**
   ```bash
   # macOS
   brew install tesseract
   # Ubuntu
   sudo apt-get install tesseract-ocr tesseract-ocr-jpn
   ```

   **libmagic (ファイルタイプ検出用)**
   ```bash
   # macOS
   brew install libmagic
   # Ubuntu
   sudo apt-get install libmagic1
   ```

4. Python依存関係をインストール

   ```bash
   pip install -r requirements.txt
   ```

5. 環境変数を設定（`.env`ファイル）

   ```env
   # AWS設定
   AWS_REGION=ap-northeast-1

   # S3設定（オプション）
   S3_BUCKET_NAME=your-bucket-name

   # モデル設定（デフォルトモデルID。Geminiもここで指定可能）
   BEDROCK_MODEL_ID=gemini-2.5-flash

   # Google Gemini設定
   GOOGLE_API_KEY=your-google-ai-api-key
   ```

## 使用方法

1. アプリケーションを起動

   ```bash
   streamlit run main.py
   ```

2. ブラウザで表示されるStreamlitインターフェースで操作
   - **モデル設定タブ**: LLMモデルとパラメータを選択
   - **検索設定タブ**: 検索文書数とフィルタを設定。画像アップロードもこちらから。
   - **ナレッジ管理タブ**: ドキュメントのアップロードや、データベースのクリアが可能。
   - **メイン画面**: 質問を入力して回答を取得。回答後にGood/Badボタンでフィードバックが可能。

## 利用可能なモデル

- **Google Gemini** (デフォルト):
  - Gemini 2.5 Flash（マルチモーダル対応）
- **AWS Bedrock**:
  - Claude 3 Haiku
  - Amazon Nova Lite

## プロジェクト構造

```text
├── main.py                 # Streamlitアプリケーションのメインエントリーポイント
├── batch_ingest.py         # バッチドキュメント取り込みスクリプト
├── src/
│   ├── aws_utils.py        # AWS BedrockとGeminiのクライアント
│   ├── document_loader.py  # ドキュメント読み込み・チャンキング
│   ├── embeddings.py       # 埋め込みモデル管理（BAAI/bge-m3）
│   ├── image_processor.py  # 画像処理（OCR + 説明生成）
│   ├── rag_chain.py        # RAGチェーン実装
│   ├── vector_store.py     # Chromaベクトルストア管理
│   └── agents/             # エージェント機能（現在無効）
├── documents/              # ドキュメント格納ディレクトリ
├── requirements.txt        # Python依存関係
├── .env                    # 環境変数設定
├── feedbacks.json          # フィードバック記録（自動生成）
└── README.md              # このファイル
```

## 機能詳細

### ドキュメント処理

- **複数フォーマット対応**: PDF, TXT, DOCX, XLSX, JPG, PNG など
- **構造保持チャンキング**: Unstructuredライブラリで文書構造を保持
- **表処理**: PDF表のMarkdown変換 + LLM要約
- **差分更新**: SQLRecordManagerによる効率的な同期

### 画像処理

- **OCR**: Tesseractによるテキスト抽出
- **マルチモーダル**: Geminiによる画像理解（オプション）
- **自動統合**: 画像をDocumentとしてベクトル化

### 検索機能

- **ハイブリッド検索**: BM25 (40%) + ベクトル検索 (60%)
- **メタデータフィルタ**: ファイルタイプ、フォルダで検索絞り込み
- **関連度スコア**: 閾値によるフィルタリング

### UI機能

- **タブ分けインターフェース**: モデル設定、検索設定、ナレッジ管理
- **リアルタイム調整**: パラメータの動的変更
- **フィードバックシステム**: 回答のGood/Bad評価
- **データベース管理**: ベクトルストアの全削除機能

## 技術仕様

### チャンキング

- **PDF/テキスト/Office**: UnstructuredLoader を使用し、`chunk_by_title` 戦略で構造を保持しながらチャンキング
- **パラメータ**: max_characters=2000, new_after_n_chars=1500
- **セパレータ**: 段落・文・単語の階層的分割

### 埋め込み

- **モデル**: BAAI/bge-m3 (多言語対応)
- **次元**: 1024
- **正規化**: 有効

### ベクトルストア

- **エンジン**: ChromaDB
- **永続化**: ローカルファイル
- **ハイブリッド**: EnsembleRetriever (BM25 + Vector)

## 注意事項

- Google Geminiを使用するにはAPIキーが必要です
- 画像処理にはTesseract OCRが必要です
- 大量のドキュメントを処理する場合、メモリ使用量に注意してください
- ChromaDBはローカル永続化のため、環境移行時はデータ移行が必要です

## 拡張性

- **新しいLLM**: BedrockClient/GeminiClientの拡張で対応可能
- **カスタムチャンキング**: DocumentLoaderの拡張で実装
- **追加モダリティ**: 音声・動画処理の追加可能

## ライセンス

このプロジェクトは学習・デモ目的で作成されたオープンソースアプリケーションです。
