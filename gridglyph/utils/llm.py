import os
from dotenv import load_dotenv
import requests
import logging
from typing import Optional, Union, Any, Dict
from abc import ABC, abstractmethod
from PIL import Image
import requests
import base64
from io import BytesIO
import logging
import base64
import torch
from transformers import AutoModel, AutoTokenizer
from urllib.request import urlopen

load_dotenv()


class LLMClient(ABC):
    """
    Abstract base class for language model clients.
    
    Supports both text-only and vision-language models.
    All clients must implement the __call__ method.
    """

    def __init__(self, *args, **kwargs):
        """
        Generic constructor to allow flexible initialization.
        
        Subclasses should extract required args/kwargs in _init_impl().
        """
        self._config = kwargs.copy()
        self._init_impl(*args, **kwargs)


    @abstractmethod
    def __call__(
        self,
        prompt: str,
        image: Optional[Union[str, Image.Image]] = None,
        **kwargs
    ) -> str:
        """
        Call the model with a prompt and optional image input.

        Args:
            prompt (str): The main instruction or question.
            image (str or PIL.Image): Optional image input.
            **kwargs: Additional parameters (e.g., system_message, temperature).

        Returns:
            str: Generated response from the model.
        """
        pass

    def get_config(self) -> dict:
        """Return the raw config used to initialize the client."""
        return self._config


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

    def _encode_image(self, image: Union[str, Image.Image]) -> str:
        if isinstance(image, str):
            if image.startswith(("http://", "https://")): 
                return image  # Use URL directly
            else:
                # Local file path → encode as base64
                with open(image, "rb") as img_file:
                    return f"data:image/png;base64,{base64.b64encode(img_file.read()).decode('utf-8')}"
        elif isinstance(image, Image.Image):
            # PIL Image object → encode in memory
            buffered = BytesIO()
            image.save(buffered, format=image.format or "PNG")
            return f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"
        else:
            raise ValueError("Image must be a path, URL, or PIL Image object")

    def __call__(
        self, prompt: str,
        image: Optional[Union[str, Image.Image]] = None,
        system_message: str = None
        ) -> str:
        """
        Generate text using the configured language model.
        
        Args:
            prompt (str): The main prompt/question to send to the model.
            image (Optional[Union[str, Image.Image]]): Optional image input (e.g., visual puzzle).
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
        
        if image is not None:
            encoded_image = self._encode_image(image)
            content = [
                {"type": "image_url", "image_url": {"url": encoded_image}},
                {"type": "text", "text": prompt}
            ]
            messages.append({"role": "user", "content": content})
        else:
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


class LocalMiniCPMVClient(LLMClient):
    """
    Local client for MiniCPM-V 2.6 model using HuggingFace Transformers.
    
    Supports both text and image inputs.
    Designed to work with ROCm (HIP) when available.
    """

    def __init__(
        self,
        model_name: str = "openbmb/MiniCPM-Llama3-V-2_5",
        device: str = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.8,
        repetition_penalty: float = 1.0,
        logger=None,
        **kwargs
    ):
        """
        Initialize the local MiniCPM-V model.

        Args:
            model_name (str): HuggingFace model identifier.
            device (str): Device to use ('cuda', 'cpu', or 'mps'). Defaults to auto-detect.
            max_tokens (int): Max tokens to generate.
            temperature (float): Sampling temperature.
            top_p (float): Nucleus sampling threshold.
            repetition_penalty (float): Penalty for repeated sequences.
            logger: Optional logger instance.
        """
        super().__init__(**kwargs)
        self.logger = logger or logging.getLogger(__name__)
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty

        # Set device
        if device:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            elif hasattr(torch, "is_hip_available") and torch.is_hip_available():
                self.device = "cuda"  # ROCm uses HIP but reports as CUDA
            else:
                self.device = "cpu"

        self.logger.info(f"Using device: {self.device}")

        try:
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = (
                AutoModel.from_pretrained(model_name, trust_remote_code=True)
                .eval()
                .to(self.device)
            )
            self.logger.info("MiniCPM-V model loaded successfully.")
        except Exception as e:
            self.logger.error(f"Error loading MiniCPM-V model: {e}")
            raise

    def _load_image(self, image_input: Union[str, Image.Image]) -> Image.Image:
        """Load and return PIL image from path, URL, or object."""
        if isinstance(image_input, str):
            if image_input.startswith(("http://", "https://")): 
                return Image.open(urlopen(image_input)).convert("RGB")
            else:
                return Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            return image_input.convert("RGB")
        else:
            raise ValueError("Image must be a path, URL, or PIL Image object")

    def __call__(
        self,
        prompt: str,
        image: Optional[Union[str, Image.Image]] = None,
        system_message: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate response from MiniCPM-V given text and/or image.

        Args:
            prompt (str): User question or instruction.
            image (Optional): Image input.
            system_message (Optional[str]): System-level instruction/context.

        Returns:
            str: Generated response.
        """
        try:
            # Prepare history
            history = []
            if system_message:
                history.append({"role": "system", "content": system_message})

            # Handle image input
            if image is not None:
                image_obj = self._load_image(image)
                messages = [
                    {"role": "user", "content": [image_obj, prompt]}  # Multimodal message
                ]
            else:
                messages = [{"role": "user", "content": prompt}]

            # Generate response
            response = self.model.chat(
                image=image_obj if image is not None else None,
                msgs=messages,
                tokenizer=self.tokenizer,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty
            )

            return response

        except Exception as e:
            self.logger.error(f"Error during generation: {e}")
            return f"[ERROR] Failed to generate response: {e}"