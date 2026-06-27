from dataclasses import dataclass

from app.core.config import Settings
from app.services.llm_service import LLMService


@dataclass
class ServicesContainer:
    settings: Settings
    llm_service: LLMService
