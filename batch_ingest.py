#!/usr/bin/env python3
"""
ベクトルDBへのドキュメント一括取り込みバッチスクリプト

指定されたディレクトリ内の全ファイルをベクトルDBに取り込みます。
汎用性を考慮して、ディレクトリパスをコマンドライン引数で指定可能。

使用例:
    python batch_ingest.py --directory /Users/mikkatagiri/Downloads
    python batch_ingest.py  # デフォルトで ~/Downloads を使用
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

try:
    import magic
    HAS_MAGIC = True
except ImportError:
    HAS_MAGIC = False
    logging.warning("python-magicがインストールされていないため、ファイルタイプ検出が制限されます。")

# プロジェクト内のモジュールをインポート
from src.document_loader import DocumentLoader
from src.embeddings import EmbeddingManager
from src.vector_store import VectorStoreManager
from src.aws_utils import BedrockClient

def setup_logging():
    """ログ設定"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('batch_ingest.log', encoding='utf-8')
        ]
    )

def validate_directory(directory_path):
    """ディレクトリの存在とアクセス権限を検証"""
    if not os.path.exists(directory_path):
        raise ValueError(f"指定されたディレクトリが存在しません: {directory_path}")

    if not os.path.isdir(directory_path):
        raise ValueError(f"指定されたパスはディレクトリではありません: {directory_path}")

    if not os.access(directory_path, os.R_OK):
        raise ValueError(f"ディレクトリへの読み取り権限がありません: {directory_path}")

    return True

def count_supported_files(directory_path):
    """サポートされているファイル数をカウント"""
    supported_extensions = {'.pdf', '.txt', '.docx', '.xlsx'}
    count = 0

    for file_path in Path(directory_path).rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            count += 1

    return count

def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description='ベクトルDBへのドキュメント一括取り込み',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python batch_ingest.py --directory /Users/mikkatagiri/Downloads
  python batch_ingest.py  # デフォルトで ~/Downloads を使用

サポートされるファイル形式: PDF, TXT, DOCX, XLSX
        """
    )

    parser.add_argument(
        '--directory',
        '-d',
        type=str,
        default=os.path.expanduser('~/Downloads'),
        help='取り込み対象のディレクトリパス (デフォルト: ~/Downloads)'
    )

    parser.add_argument(
        '--persist-dir',
        type=str,
        default='./chroma_db',
        help='ベクトルストアの保存ディレクトリ (デフォルト: ./chroma_db)'
    )

    parser.add_argument(
        '--force-recreate',
        action='store_true',
        help='既存のベクトルストアを強制的に再作成'
    )

    args = parser.parse_args()

    # ログ設定
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        # 環境変数読み込み
        load_dotenv()

        logger.info("="*60)
        logger.info("🚀 ベクトルDB一括取り込みバッチを開始")
        logger.info("="*60)
        logger.info(f"対象ディレクトリ: {args.directory}")
        logger.info(f"ベクトルストア保存先: {args.persist_dir}")

        # ディレクトリ検証
        validate_directory(args.directory)

        # サポートファイル数の確認
        file_count = count_supported_files(args.directory)
        if file_count == 0:
            logger.warning(f"指定ディレクトリにサポートされるファイルが見つかりません: {args.directory}")
            logger.info("サポート形式: PDF, TXT, DOCX, XLSX, MD, PY, JS, HTML, CSV など")
            if not HAS_MAGIC:
                logger.info("ヒント: python-magicをインストールすると、より正確なファイルタイプ検出が可能です")
            return

        logger.info(f"見つかったサポートファイル数: {file_count}")

        # 1. 埋め込みモデル準備
        logger.info("\n🤖 ステップ1: 埋め込みモデル準備")
        embedding_manager = EmbeddingManager()
        embeddings = embedding_manager.get_embeddings()

        # 2. ベクトルストア準備
        logger.info("\n💾 ステップ2: ベクトルストア準備")
        vector_manager = VectorStoreManager(embeddings, persist_directory=args.persist_dir)

        # 既存ベクトルストアの処理
        vectorstore_exists = os.path.exists(args.persist_dir)

        if args.force_recreate and vectorstore_exists:
            logger.info("既存のベクトルストアを強制再作成します")
            import shutil
            shutil.rmtree(args.persist_dir)
            vectorstore_exists = False

        if vectorstore_exists:
            try:
                vector_manager.load_vectorstore()
                logger.info("✅ 既存のベクトルストアを読み込みました")
            except Exception as e:
                logger.warning(f"既存ベクトルストアの読み込みに失敗しました: {e}")
                logger.info("新規ベクトルストアを作成します")
                vectorstore_exists = False

        if not vectorstore_exists:
            logger.info("新規ベクトルストアを作成します")

        # 3. Bedrockクライアント準備（オプション）
        bedrock_client = None
        try:
            bedrock_client = BedrockClient(region_name=os.getenv('AWS_REGION', 'ap-northeast-1'))
            logger.info("✅ AWS Bedrockクライアントを初期化しました")
        except Exception as e:
            logger.warning(f"AWS Bedrockクライアントの初期化に失敗しました: {e}")
            logger.info("表処理機能が無効になります")

        # 4. ドキュメント読み込み
        logger.info("\n📄 ステップ3: ドキュメント読み込み")
        loader = DocumentLoader(bedrock_client=bedrock_client)
        documents = loader.load_directory(args.directory)

        if not documents:
            logger.error("ドキュメントの読み込みに失敗しました")
            return

        logger.info(f"✅ {len(documents)}個のチャンクを作成しました")

        # 5. ベクトルストアへの同期
        logger.info("\n🔄 ステップ4: ベクトルストアへの同期")
        if not vectorstore_exists:
            # 新規作成の場合
            vector_manager.create_vectorstore(documents)
            logger.info("✅ 新規ベクトルストアの作成が完了しました")
        else:
            # 既存への追加
            result = vector_manager.sync_documents(documents)
            logger.info("✅ ベクトルストアの同期が完了しました")
            logger.info(f"結果: 追加 {result['num_added']}, 更新 {result['num_updated']}, スキップ {result['num_skipped']}, 削除 {result['num_deleted']}")

        logger.info("\n🎉 一括取り込み処理が完了しました！")
        logger.info(f"処理したドキュメント数: {len(documents)}")
        logger.info(f"ベクトルストア保存先: {args.persist_dir}")

    except Exception as e:
        logger.error(f"バッチ処理中にエラーが発生しました: {e}")
        logger.error("詳細なエラー情報:", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()