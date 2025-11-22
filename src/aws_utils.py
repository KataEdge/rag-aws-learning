import boto3
import time
import json
from typing import Dict, Any
import botocore
import random

# リクエスト間に2秒待機
time.sleep(2)

import os

class BedrockClient:
    def __init__(self, region_name='ap-northeast-1', model_id=None):
        self.client = boto3.client(
            service_name='bedrock-runtime',
            region_name=region_name
        )
        if model_id:
            self.model_id = model_id
        else:
            self.model_id = os.getenv('BEDROCK_MODEL_ID', 'anthropic.claude-3-haiku-20240307-v1:0')
    
    def invoke_claude(self, prompt: str, max_tokens: int = 2000) -> str:
        """
        Claude 3 Haikuを呼び出し
        """
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.7,
        })
        
        max_retries = 5
        backoff = 0.5
        
        for attempt in range(max_retries):
            try:
                response = self.client.invoke_model(
                    modelId=self.model_id,
                    body=body
                )
                
                response_body = json.loads(response['body'].read())
                return response_body['content'][0]['text']
            
            except botocore.exceptions.ClientError as e:
                error_code = e.response['Error']['Code']
                if error_code == 'ThrottlingException':
                    wait = backoff * (2 ** attempt) + random.uniform(0, 0.5)
                    print(f"⚠️ Throttling発生。{wait:.1f}秒待機して再試行します ({attempt+1}/{max_retries})")
                    time.sleep(wait)
                else:
                    raise
            except Exception as e:
                print(f"❌ Bedrock呼び出しエラー: {e}")
                raise
    
    def test_connection(self) -> bool:
        """
        接続テスト
        """
        try:
            result = self.invoke_claude("Hello, please respond with 'OK'")
            print(f"✅ Bedrock接続成功: {result}")
            return True
        except Exception as e:
            print(f"❌ Bedrock接続失敗: {e}")
            return False