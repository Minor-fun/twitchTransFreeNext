#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ai_translator.py
AI翻訳エンジン（OpenAI互換API）

OpenAI Chat Completions API互換のサービスをサポート：
- OpenAI (api.openai.com)
- OpenRouter (openrouter.ai)
- Cloudflare AI Gateway
- ローカルLLM (Ollama, LM Studio など)
"""

import aiohttp
import asyncio
from urllib.parse import urlparse, urlunparse

# デフォルトシステムプロンプト
DEFAULT_SYSTEM_PROMPT = """You are a translator for Twitch chat messages.
Your task is to translate the given text accurately and naturally.
Rules:
1. Output ONLY the translated text, nothing else.
2. Do not add explanations, quotes, or prefixes.
3. Keep the tone casual and natural, as it's chat messages.
4. Preserve emotes, usernames, and special formatting."""


class AITranslator:
    """AI翻訳エンジン（OpenAI互換API）"""

    def __init__(self, api_url: str, api_key: str, model: str,
                 system_prompt: str = None, temperature: float = 0.3, debug: bool = False):
        """
        AI翻訳器を初期化

        Args:
            api_url: APIエンドポイントURL
            api_key: APIキー
            model: モデル名
            system_prompt: カスタムシステムプロンプト（省略可、Noneの場合はデフォルトを使用）
            temperature: 温度パラメータ（0.0-2.0）
            debug: デバッグ出力を有効にするかどうか
        """
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.system_prompt = system_prompt if system_prompt else DEFAULT_SYSTEM_PROMPT
        self.temperature = temperature
        self.debug = debug


    def _get_api_url(self) -> str:
        """
        完全なAPI URLを取得し、/chat/completionsパスを自動補完

        Returns:
            完全なAPI URL
        """
        target_url = self.api_url

        try:
            parsed = urlparse(self.api_url)
            path = parsed.path

            # パスが /chat/completions で終わっていない場合、自動補完
            if not path.endswith('/chat/completions'):
                if path.endswith('/'):
                    path = path + 'chat/completions'
                else:
                    path = path + '/chat/completions'

                target_url = urlunparse((
                    parsed.scheme,
                    parsed.netloc,
                    path,
                    parsed.params,
                    parsed.query,
                    parsed.fragment
                ))
        except Exception:
            # URL解析失敗、リクエストを自然に失敗させる
            pass

        return target_url


    def _build_request_body(self, text: str, source_lang: str, target_lang: str) -> dict:
        """
        リクエストボディを構築

        Args:
            text: 翻訳するテキスト
            source_lang: ソース言語
            target_lang: ターゲット言語

        Returns:
            OpenAI Chat Completions形式のリクエストボディ
        """
        user_message = f"""Source Language: {source_lang}
Target Language: {target_lang}
Text to translate:
\"\"\"
{text}
\"\"\"
Translation:"""

        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_message}
            ],
            "temperature": self.temperature
        }


    async def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        翻訳を実行

        Args:
            text: 翻訳するテキスト
            source_lang: ソース言語コード
            target_lang: ターゲット言語コード

        Returns:
            翻訳されたテキスト、失敗時は空文字列
        """
        if not text:
            return ''

        api_url = self._get_api_url()
        request_body = self._build_request_body(text, source_lang, target_lang)

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    api_url,
                    json=request_body,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 401:
                        if self.debug:
                            print('[AI Translator] Invalid API Key')
                        return ''

                    if response.status == 429:
                        if self.debug:
                            print('[AI Translator] Rate limit exceeded')
                        return ''

                    if response.status >= 500:
                        if self.debug:
                            print(f'[AI Translator] API server error: {response.status}')
                        return ''

                    if not response.ok:
                        error_text = await response.text()
                        if self.debug:
                            print(f'[AI Translator] API error: {response.status} - {error_text}')
                        return ''

                    data = await response.json()
                    translated_text = data.get('choices', [{}])[0].get('message', {}).get('content', '')

                    if not translated_text:
                        if self.debug:
                            print('[AI Translator] Empty response from API')
                        return ''

                    return translated_text.strip()

        except asyncio.TimeoutError:
            if self.debug:
                print('[AI Translator] API request timeout')
            return ''

        except aiohttp.ClientError as e:
            if self.debug:
                print(f'[AI Translator] Network error: {e}')
            return ''

        except Exception as e:
            if self.debug:
                print(f'[AI Translator] Unexpected error: {e}')
            return ''



# グローバル翻訳器インスタンス
_translator: AITranslator = None


def init_translator(api_url: str, api_key: str, model: str,
                    system_prompt: str = None, temperature: float = 0.3, debug: bool = False):
    """
    グローバル翻訳器インスタンスを初期化

    Args:
        api_url: APIエンドポイントURL
        api_key: APIキー
        model: モデル名
        system_prompt: カスタムシステムプロンプト（省略可）
        temperature: 温度パラメータ（0.0-2.0）
        debug: デバッグ出力を有効にするかどうか
    """
    global _translator
    _translator = AITranslator(api_url, api_key, model, system_prompt, temperature, debug)


async def translate(text: str, source_lang: str, target_lang: str) -> str:
    """
    グローバル翻訳器を使用して翻訳を実行

    Args:
        text: 翻訳するテキスト
        source_lang: ソース言語コード
        target_lang: ターゲット言語コード

    Returns:
        翻訳されたテキスト、失敗時は空文字列
    """
    global _translator
    if _translator is None:
        print('[AI Translator] Translator not initialized. Call init_translator() first.')
        return ''
    return await _translator.translate(text, source_lang, target_lang)
