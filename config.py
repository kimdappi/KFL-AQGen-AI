"""
Configuration settings for the Korean worksheet generator
"""
import os
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class Config:
    # Data paths
    PDFS_DIR: str = "data/pdfs"
    SCHEMAS_DIR: str = "data/schemas"
    
    # Difficulty levels
    DIFFICULTY_LEVELS: Dict[str, str] = None
    
    # MCP settings (for K-pop agent)
    MCP_ENABLED: bool = True
    MCP_TIMEOUT: int = 30
    
    # Worksheet generation settings
    MAX_ITEMS_PER_SECTION: int = 10
    MIN_ITEMS_PER_WORKSHEET: int = 3
    DEFAULT_TIME_ESTIMATE: str = "30-45 minutes"
    
    # Validation settings
    MIN_VALIDATION_SCORE: float = 60.0
    TARGET_VALIDATION_SCORE: float = 80.0
    
    def __post_init__(self):
        if self.DIFFICULTY_LEVELS is None:
            self.DIFFICULTY_LEVELS = {
                "easy": "data/pdfs/easy",
                "medium": "data/pdfs/medium", 
                "hard": "data/pdfs/hard"
            }
    
    @classmethod
    def from_env(cls) -> 'Config':
        """Load configuration from environment variables"""
        return cls(
            PDFS_DIR=os.getenv("PDFS_DIR", "data/pdfs"),
            SCHEMAS_DIR=os.getenv("SCHEMAS_DIR", "data/schemas"),
            MCP_ENABLED=os.getenv("MCP_ENABLED", "true").lower() == "true",
            MCP_TIMEOUT=int(os.getenv("MCP_TIMEOUT", "30")),
            MAX_ITEMS_PER_SECTION=int(os.getenv("MAX_ITEMS_PER_SECTION", "10")),
            MIN_ITEMS_PER_WORKSHEET=int(os.getenv("MIN_ITEMS_PER_WORKSHEET", "3")),
            MIN_VALIDATION_SCORE=float(os.getenv("MIN_VALIDATION_SCORE", "60.0")),
            TARGET_VALIDATION_SCORE=float(os.getenv("TARGET_VALIDATION_SCORE", "80.0"))
        )


# Global config instance
config = Config.from_env()
