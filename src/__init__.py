"""
Gemini Deep Research Agent Package
=================================

A custom implementation of Google's Deep Research functionality.
"""

__version__ = "1.0.0"
__author__ = "Deep Research Team"

from .deep_research_agent import (
    DeepResearchAgent,
    TokenCounter,
    ResearchConfig,
    TokenUsage
)

__all__ = [
    "DeepResearchAgent",
    "TokenCounter", 
    "ResearchConfig",
    "TokenUsage"
]