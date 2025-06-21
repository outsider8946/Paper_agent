import os
from omegaconf import DictConfig
from typing import Optional
from dotenv import load_dotenv
from pydantic import Field, SecretStr
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.utils.utils import secret_from_env

load_dotenv()

class LLMOlama():
    def __init__(self, config: DictConfig):
        self.llm = ChatOllama(model=config.llm.model_name,
                              temperature=config.llm.temperature,
                              top_k=config.llm.top_k,
                              top_p=config.llm.top_p,
                              repeat_penalty=config.llm.repeat_penalty)

class LLMOpenRouter(ChatOpenAI):
    api: Optional[SecretStr] = Field(
        alias="api_key",
        default_factory=secret_from_env("OPENROUTER_API_KEY", default=None),
    )

    @property
    def lc_secrets(self) -> dict[str, str]:
        return {"api_key": "OPENROUTER_API_KEY"}

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        api_key = (api_key or os.environ.get("OPENROUTER_API_KEY"))
        super().__init__(base_url="https://openrouter.ai/api/v1", api_key=api_key, **kwargs)