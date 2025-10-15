import boto3
import os
from pathlib import Path

class S3Manager:
    def __init__(self, bucket_name: str, region_name: str = 'ap-northeast-1'):
        self.s3_client = boto3.client('s3', region_name=region_name)
        self.bucket_name = bucket_name
        self.region_name = region_name
    
    def create_bucket_if_not_exists(self):
        """
        S3バケットが存在しなければ作成
        """
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            print(f"✅ バケット '{self.bucket_name}' は既に存在します")
        except:
            try:
                if self.region_name == 'us-east-1':
                    self.s3_client.create_bucket(Bucket=self.bucket_name)
                else:
                    self.s3_client.create_bucket(
                        Bucket=self.bucket_name,
                        CreateBucketConfiguration={'LocationConstraint': self.region_name}
                    )
                print(f"✅ バケット '{self.bucket_name}' を作成しました")
            except Exception as e:
                print(f"❌ バケット作成エラー: {e}")
                raise
    
    def upload_file(self, file_path: str, s3_key: str = None):
        """
        ファイルをS3にアップロード
        """
        if s3_key is None:
            s3_key = Path(file_path).name
        
        try:
            self.s3_client.upload_file(file_path, self.bucket_name, s3_key)
            print(f"✅ アップロード完了: {file_path} → s3://{self.bucket_name}/{s3_key}")
            return s3_key
        except Exception as e:
            print(f"❌ アップロードエラー: {e}")
            raise
    
    def download_file(self, s3_key: str, local_path: str):
        """
        S3からファイルをダウンロード
        """
        try:
            self.s3_client.download_file(self.bucket_name, s3_key, local_path)
            print(f"✅ ダウンロード完了: s3://{self.bucket_name}/{s3_key} → {local_path}")
            return local_path
        except Exception as e:
            print(f"❌ ダウンロードエラー: {e}")
            raise
    
    def list_files(self, prefix: str = ''):
        """
        バケット内のファイル一覧を取得
        """
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            
            if 'Contents' in response:
                files = [obj['Key'] for obj in response['Contents']]
                return files
            else:
                return []
        except Exception as e:
            print(f"❌ ファイル一覧取得エラー: {e}")
            raise