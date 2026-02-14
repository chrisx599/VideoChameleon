import logging
from pydantic import BaseModel
from typing import Optional, List, Any

from univa.utils.logging_setup import configure_logging

class ToolResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    content: Optional[Any] = None
    output_path: Optional[str|List[str]] = None
    
    class Config:
        extra = "allow"


def setup_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)
