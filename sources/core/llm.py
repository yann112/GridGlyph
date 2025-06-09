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
    """
    OpenRouter API client for language model interactions.
    
    This client provides a stable interface for generating text using various language models
    through the OpenRouter API, with configurable parameters for controlling output consistency
    and quality.
    """
    
    def __init__(
        self,
        logger=None,
        api_key=None,
        model="mistralai/mistral-small-3.1-24b-instruct",
        infra="deepinfra",
        max_tokens=1200,
        proxy: str = None,
        temperature=0.0,
        top_p=0.1,
        top_k=1,
        repetition_penalty=1.0,
        stream=False,
        seed=None,
    ):
        """
        Initialize the OpenRouter client.
        
        Args:
            logger: Logger instance for error reporting. Defaults to module logger.
            api_key (str, optional): OpenRouter API key. If not provided, will use 
                OPEN_ROUTER_API_KEY environment variable.
            model (str): Model identifier to use. Default is Mistral 24B instruct model.
            max_tokens (int): Maximum number of tokens to generate in response.
            proxy (str, optional): HTTP/HTTPS proxy URL if needed.
            temperature (float): Controls randomness in output. Range 0.0-2.0.
                - 0.0: Completely deterministic (same input = same output)
                - 1.0: Default balanced randomness
                - 2.0: Maximum creativity/randomness
            top_p (float): Nucleus sampling parameter. Range 0.0-1.0.
                Controls diversity by limiting token choices to top P probability mass.
                Lower values = more focused/predictable output.
            top_k (int): Limits token choices to top K most likely tokens.
                - 1: Always pick most likely token (very deterministic)
                - 0: No limit (consider all tokens)
            repetition_penalty (float): Reduces repetition of input tokens. Range 0.0-2.0.
                - 1.0: No penalty (neutral)
                - >1.0: Discourage repetition
                - <1.0: Encourage repetition
            stream (bool): Whether to stream response. False for complete response.
            seed (int, optional): Random seed for deterministic sampling.
                Same seed + same params should give same results (model dependent).
                
        Raises:
            ValueError: If no API key is provided via parameter or environment variable.
        """
        self.logger = logger or logging.getLogger(__file__)
        self.api_key = api_key or os.environ.get("OPEN_ROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("API key is required. Set it in the environment or pass it explicitly.")
        
        self.model = model
        self.infra = infra
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.stream = stream
        self.seed = seed
        
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Provider": self.infra,
        }
        
        self.proxies = None
        if proxy:
            # Set proxies dict for requests
            self.proxies = {
                "http": proxy,
                "https": proxy,
            }

    def __call__(self, prompt: str, system_message: str = None) -> str:
        """
        Generate text using the configured language model.
        
        Args:
            prompt (str): The main prompt/question to send to the model.
            system_message (str, optional): System message to set context/instructions.
                Useful for setting behavior, output format, or role definition.
                
        Returns:
            str: Generated text response from the model, or None if request fails.
            
        Example:
            >>> client = OpenRouterClient(temperature=0.0)  # Deterministic output
            >>> response = client("What is 2+2?")
            >>> print(response)
            "4"
            
            >>> # Using system message for code generation
            >>> system_msg = "You are a Python code generator. Return only valid code."
            >>> code = client("Write a function to add two numbers", system_msg)
        """
        # Build messages array
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        # Build request data with all parameters
        data = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "stream": self.stream,
        }
        
        # Add seed if specified
        if self.seed is not None:
            data["seed"] = self.seed

        try:
            response = requests.post(
                self.base_url, 
                headers=self.headers, 
                json=data, 
                proxies=self.proxies,
                timeout=30  # Add timeout for stability
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