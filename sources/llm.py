import os
from dotenv import load_dotenv
import requests
import logging
from abc import ABC, abstractmethod

load_dotenv()


class LLMClient(ABC):
    @abstractmethod
    def __call__(self, prompt: str) -> str:
        """Call the language model with a prompt and return the response."""
        pass


class OpenRouterClient(LLMClient):
    def __init__(
        self,
        logger=None,
        api_key=None,
        model="mistralai/mistral-small-3.1-24b-instruct",
        max_tokens=1200,
        proxy: str = None,
    ):
        self.logger = logger or logging.getLogger(__file__)
        self.api_key = api_key or os.environ.get("OPEN_ROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("API key is required. Set it in the environment or pass it explicitly.")
        self.model = model
        self.max_tokens = max_tokens
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        self.proxies = None
        if proxy:
            # Set proxies dict for requests
            self.proxies = {
                "http": proxy,
                "https": proxy,
            }

    def __call__(self, prompt: str) -> str:
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
        }

        try:
            response = requests.post(
                self.base_url, headers=self.headers, json=data, proxies=self.proxies
            )
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content']
        except requests.exceptions.RequestException as e:
            self.logger.error(f"HTTP error: {e}")
            return None
        except (KeyError, IndexError) as e:
            self.logger.error(f"Error parsing response: {e}, Response text: {response.text}")
            return None
