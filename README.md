# RAG AWS Learning - 多様なLLMを活用したRAGデモアプリケーション

このプロジェクトは、Retrieval-Augmented Generation (RAG) を使用したドキュメント検索・質問応答システムのデモアプリケーションです。AWS Bedrock、Google Gemini、Amazon Novaなどの複数のLLMをサポートしています。

## 特徴

- **多様なLLMサポート**: AWS Bedrock (Claude, Nova)、Google Geminiを統合
- **ドキュメント管理**: PDF、テキスト、Word、Excelファイルのアップロードと自動処理
- **ベクトル検索**: FAISSを使用した効率的な類似度検索
- **ハイブリッド検索**: キーワード検索とベクトル検索の組み合わせ
- **Streamlit UI**: 直感的なWebインターフェース
- **会話履歴**: コンテキストを保持した継続的な対話
- **動的パラメータ調整**: Temperatureや検索文書数のリアルタイム変更

## アーキテクチャ

```text
Documents → Document Loader → Embeddings → Vector Store → Retriever → LLM → Answer
```

## 必要条件

- Python 3.8+
- AWSアカウント（Bedrock利用時）
- Google AI APIキー（Gemini利用時）

## インストール

1. リポジトリをクローンまたはダウンロード

2. 仮想環境を作成・有効化

   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. 依存関係をインストール

   ```bash
   pip install -r requirements.txt
   ```

4. 環境変数を設定（`.env`ファイル）

   ```env
   # AWS設定
   AWS_REGION=ap-northeast-1

   # S3設定（オプション）
   S3_BUCKET_NAME=your-bucket-name

   # Bedrockモデル設定
   BEDROCK_MODEL_ID=anthropic.claude-3-haiku-20240307-v1:0

   # Google AI APIキー（Gemini利用時）
   GOOGLE_API_KEY=your-google-ai-api-key
   ```

## 使用方法

1. アプリケーションを起動

   ```bash
   python main.py
   ```

2. ブラウザで表示されるStreamlitインターフェースで操作
   - LLMモデルを選択
   - Temperatureと検索文書数を調整
   - ドキュメントをアップロード
   - 質問を入力して回答を取得

## 利用可能なモデル

- **AWS Bedrock**:
  - Claude 3 Haiku
  - Amazon Nova Lite/Pro
- **Google Gemini**:
  - Gemini 1.5 Flash

## プロジェクト構造

```text
├── main.py                 # Streamlitアプリケーションのメインエントリーポイント
├── batch_ingest.py         # バッチドキュメント取り込みスクリプト
├── src/
│   ├── aws_utils.py        # AWS BedrockとGeminiのクライアント
│   ├── document_loader.py  # ドキュメント読み込み・処理
│   ├── embeddings.py       # 埋め込みモデル管理
│   ├── rag_chain.py        # RAGチェーン実装
│   ├── vector_store.py     # ベクトルストア管理
│   └── s3_manager.py       # S3連携（オプション）
├── documents/              # サンプルドキュメント
├── requirements.txt        # Python依存関係
└── .env                    # 環境変数設定
```

## 機能詳細

### ドキュメント処理

- 複数フォーマット対応（PDF, TXT, DOCX, XLSX）
- 自動テキスト抽出とチャンク分割
- 差分更新による効率的な知識ベース管理

### 検索機能

- ベクトル検索による意味ベースの検索
- BM25キーワード検索とのハイブリッド検索
- 関連度スコアによるランキング

### UI機能

- リアルタイムチャットインターフェース
- モデル切り替え
- パラメータ調整スライダー
- ソース文書の表示

## 注意事項

- AWS Bedrockを使用するには適切なIAM権限が必要です
- Geminiを使用するにはGoogle AI APIキーが必要です
- 大量のドキュメントを処理する場合、メモリ使用量に注意してください

## ライセンス

このプロジェクトは学習目的で作成されたデモアプリケーションです。
