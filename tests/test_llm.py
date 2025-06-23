import pytest
from pathlib import Path
from dotenv import load_dotenv

from core.llm import OpenRouterClient

load_dotenv()


@pytest.mark.integration
def test_open_router():
    client = OpenRouterClient()

    prompt = "it is for testing purpose just answer the single word 'ready' nothing more"
    result = client(prompt)

    assert result is not None, "The API did not return a response."
    assert "ready" in result.lower(), f"Unexpected response: {result}"

@pytest.mark.integration
def test_open_router_vision():
    client = OpenRouterClient(
        model="qwen/qwen2.5-vl-72b-instruct",
        infra="deepinfra",
        max_tokens=300,
        temperature=0.0
    )
    image_path = Path('tests/data/image.png')
    response = client(
        prompt="Solve this visual logic puzzle step-by-step.",
        image=str(image_path)
    )

    assert response