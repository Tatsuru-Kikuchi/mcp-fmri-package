#!/usr/bin/env python3
"""
MCP-fMRI: Ethical Analysis of Mathematical Abilities Using Generative AI

A Python package for ethical analysis of mathematical cognition using fMRI data
with emphasis on gender similarities and cultural context.
"""

__version__ = "0.1.0"
__author__ = "Tatsuru Kikuchi"
__email__ = "contact@example.com"
__description__ = "Ethical Analysis of Mathematical Abilities Using Generative AI and fMRI"

from .preprocessing import fMRIPreprocessor
from .analysis import (
    GenderSimilarityAnalyzer,
    EthicalfMRIAnalysis,
    BiasDetector
)
from .cultural import JapaneseCulturalContext
from .visualization import (
    SimilarityPlotter,
    BrainNetworkPlotter,
    EthicalReportGenerator
)
from .utils import (
    load_config,
    validate_data,
    ethical_guidelines
)

__all__ = [
    "__version__",
    "fMRIPreprocessor",
    "GenderSimilarityAnalyzer",
    "EthicalfMRIAnalysis",
    "BiasDetector",
    "JapaneseCulturalContext",
    "SimilarityPlotter",
    "BrainNetworkPlotter",
    "EthicalReportGenerator",
    "load_config",
    "validate_data",
    "ethical_guidelines",
]

# Package-level configuration
CONFIG = {
    "ethical_guidelines": True,
    "similarity_focus": True,
    "cultural_context": "japanese",
    "bias_detection": True,
    "individual_emphasis": True,
}

def get_config():
    """Get package configuration."""
    return CONFIG.copy()

def set_config(**kwargs):
    """Update package configuration."""
    CONFIG.update(kwargs)

# Ethical reminder
ETHICAL_NOTICE = """
IMPORTANT: This package emphasizes gender similarities in mathematical cognition.
All analyses should focus on commonalities rather than differences and include
appropriate cultural context. Results should never be used to justify
discrimination or reinforce stereotypes.
"""

def print_ethical_notice():
    """Print ethical guidelines notice."""
    print(ETHICAL_NOTICE)