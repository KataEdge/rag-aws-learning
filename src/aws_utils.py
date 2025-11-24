import boto3
import time
import json
import base64
from typing import Dict, Any, List, Union
import botocore
import random
import google.generativeai as genai

# リクエスト間に2秒待機
time.sleep(2)

import os

class BedrockClient:
    def __init__(self, region_name='us-east-1', model_id=None):
        self.client = boto3.client(
            service_name='bedrock-runtime',
            region_name=region_name
        )
        if model_id:
            self.model_id = model_id
        else:
            self.model_id = os.getenv('BEDROCK_MODEL_ID', 'anthropic.claude-3-haiku-20240307-v1:0')

    def _get_model_type(self) -> str:
        """モデルIDからモデルタイプを判定"""
        if 'anthropic.claude' in self.model_id:
            return 'claude'
        elif 'amazon.nova' in self.model_id:
            return 'nova'
        elif 'amazon.titan' in self.model_id:
            return 'titan'
        elif 'meta.llama' in self.model_id:
            return 'llama'
        else:
            return 'unknown'

    def invoke_model(self, prompt: str, system_prompt: str = None, temperature: float = 0.7, max_tokens: int = 2000, images: List[bytes] = None) -> str:
        """
        汎用的なモデル呼び出しメソッド（テキスト + 画像対応）
        """
        model_type = self._get_model_type()
        print(f"🔧 モデル: {self.model_id}, Temperature: {temperature}, Max Tokens: {max_tokens}")

        if model_type == 'claude':
            # コンテンツの構築
            content = []
            if images:
                for image_bytes in images:
                    # 画像をbase64エンコード
                    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                    content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",  # 仮定、実際はMIMEタイプを検出
                            "data": image_base64
                        }
                    })
            content.append({
                "type": "text",
                "text": prompt
            })

            payload = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "messages": [
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                "temperature": temperature,
            }
            # システムプロンプトがある場合に追加
            if system_prompt:
                payload["system"] = system_prompt
            body = json.dumps(payload)
        elif model_type == 'titan':
            body = json.dumps({
                "inputText": prompt,
                "textGenerationConfig": {
                    "maxTokenCount": max_tokens,
                    "temperature": temperature,
                    "topP": 0.9
                }
            })
        elif model_type == 'nova':
            payload = {
                "inferenceConfig": {
                    "maxTokens": max_tokens,
                    "temperature": temperature,
                    "topP": 0.9
                },
                "messages": [
                    {
                        "role": "user",
                        "content": [{"text": prompt}]
                    }
                ]
            }
            if system_prompt:
                payload["system"] = [{"text": system_prompt}]
            body = json.dumps(payload)
        elif model_type == 'llama':
            # Llama 3 のチャットフォーマットを使用
            system_text = system_prompt if system_prompt is not None else "You are a helpful assistant."
            formatted_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_text}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            body = json.dumps({
                "prompt": formatted_prompt,
                "max_gen_len": max_tokens,
                "temperature": temperature,
                "top_p": 0.9
            })
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        max_retries = 5
        backoff = 0.5

        for attempt in range(max_retries):
            try:
                response = self.client.invoke_model(
                    modelId=self.model_id,
                    body=body
                )

                response_body = json.loads(response['body'].read())

                if model_type == 'claude':
                    return response_body['content'][0]['text']
                elif model_type == 'nova':
                    return response_body['output']['message']['content'][0]['text']
                elif model_type == 'titan':
                    return response_body['results'][0]['outputText']
                elif model_type == 'llama':
                    return response_body['generation']
                else:
                    raise ValueError(f"Unsupported model type: {model_type}")

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
    
    def invoke_claude(self, prompt: str, system_prompt: str = None, temperature: float = 0.7, max_tokens: int = 2000, images: List[bytes] = None) -> str:
        """
        Claudeモデルを呼び出し（後方互換性のため）
        """
        return self.invoke_model(prompt, system_prompt, temperature, max_tokens, images)
    
    def test_connection(self) -> bool:
        """
        接続テスト
        """
        try:
            result = self.invoke_model("Hello, please respond with 'OK'", temperature=0.1)
            print(f"✅ Bedrock接続成功: {result}")
            return True
        except Exception as e:
            print(f"❌ Bedrock接続失敗: {e}")
            return False


class GeminiClient:
    def __init__(self, model_name='gemini-1.5-flash'):
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set")
        genai.configure(api_key=api_key)
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)

    def invoke_model(self, prompt: str, system_prompt: str = None, temperature: float = 0.7, max_tokens: int = 2000, images: List[bytes] = None) -> str:
        """
        Geminiモデルを呼び出し（マルチモーダル対応）
        """
        print(f"🔧 Geminiモデル: {self.model_name}, Temperature: {temperature}, Max Tokens: {max_tokens}")

        # Geminiではシステムプロンプトを最初のメッセージとして扱う
        messages = []
        if system_prompt:
            messages.append({"role": "user", "parts": [system_prompt]})
            messages.append({"role": "model", "parts": ["了解しました。"]})  # システムプロンプトの確認

        # ユーザー入力の構築
        user_parts = []
        if images:
            for image_bytes in images:
                # PIL Imageに変換
                from PIL import Image
                import io
                image = Image.open(io.BytesIO(image_bytes))
                user_parts.append(image)
        user_parts.append(prompt)
        messages.append({"role": "user", "parts": user_parts})

        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

        try:
            response = self.model.generate_content(
                messages,
                generation_config=generation_config
            )

            # 安全フィルターのチェック
            if response.candidates:
                candidate = response.candidates[0]
                if candidate.finish_reason == 2:  # SAFETY
                    return "申し訳ありませんが、安全ポリシーに違反する内容のため、回答を生成できませんでした。"
                elif candidate.finish_reason != 1:  # 1 is FINISH_REASON_UNSPECIFIED (normal)
                    return f"応答が完了しませんでした。理由: {candidate.finish_reason}"

            return response.text
        except Exception as e:
            print(f"❌ Gemini呼び出しエラー: {e}")
            raise

    def invoke_claude(self, prompt: str, system_prompt: str = None, temperature: float = 0.7, max_tokens: int = 2000, images: List[bytes] = None) -> str:
        """
        Claude互換メソッド（後方互換性のため）
        """
        return self.invoke_model(prompt, system_prompt, temperature, max_tokens, images)

    def test_connection(self) -> bool:
        """
        接続テスト
        """
        try:
            result = self.invoke_model("Hello, please respond with 'OK'", temperature=0.1)
            print(f"✅ Gemini接続成功: {result}")
            return True
        except Exception as e:
            print(f"❌ Gemini接続失敗: {e}")
            return False