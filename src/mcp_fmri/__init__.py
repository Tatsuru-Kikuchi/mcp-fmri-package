#!/usr/bin/env python3
"""
MCP-fMRI: Ethical Analysis of Mathematical Abilities Using Generative AI

A Python package for ethical analysis of mathematical cognition using fMRI data
with emphasis on gender similarities and cultural context.

Now includes comprehensive economic bias detection and mitigation tools
specifically designed for economic research applications.
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
from .economic_bias import (
    EconomicBiasDetector,
    run_comprehensive_economic_bias_analysis
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
    "EconomicBiasDetector",
    "run_comprehensive_economic_bias_analysis",
]

# Package-level configuration
CONFIG = {
    "ethical_guidelines": True,
    "similarity_focus": True,
    "cultural_context": "japanese",
    "bias_detection": True,
    "individual_emphasis": True,
    "economic_bias_detection": True,  # New feature
}

def get_config():
    """Get package configuration."""
    return CONFIG.copy()

def set_config(**kwargs):
    """Update package configuration."""
    CONFIG.update(kwargs)

# Ethical reminder with economic considerations
ETHICAL_NOTICE = """
IMPORTANT: This package emphasizes gender similarities in mathematical cognition.
All analyses should focus on commonalities rather than differences and include
appropriate cultural context. Results should never be used to justify
discrimination or reinforce stereotypes.

ECONOMIC RESEARCH NOTICE: Special attention to economic bias is critical.
Economic factors (SES, income, education access) can confound neuroimaging
results and lead to incorrect policy conclusions. Use comprehensive bias
detection tools before drawing economic implications.

REMEMBER: Individual differences exceed group differences. Focus on individual
potential and structural factors rather than group-based generalizations.
"""

def print_ethical_notice():
    """Print ethical guidelines notice with economic considerations."""
    print(ETHICAL_NOTICE)

def print_economic_bias_warning():
    """Print specific warning about economic bias in research."""
    warning = """
⚠️  ECONOMIC BIAS WARNING ⚠️

Economic factors are major sources of bias in neuroimaging research:

1. SELECTION BIAS: Non-representative sampling by income/education
2. SES CONFOUNDING: Socioeconomic status affects both brain and behavior
3. TEMPORAL BIAS: Economic conditions during data collection
4. ALGORITHMIC BIAS: Models may discriminate by economic factors
5. POLICY MISUSE: Findings used to justify economic discrimination

USE MCP-fMRI's economic bias detection tools BEFORE drawing conclusions!

Example:
    from mcp_fmri.economic_bias import EconomicBiasDetector
    detector = EconomicBiasDetector(sensitivity='high')
    bias_results = detector.detect_selection_bias_economic(demographics)
    """
    print(warning)

# Quick bias check function
def quick_economic_bias_check(demographics_data, brain_data=None, performance_data=None):
    """Quick economic bias check for researchers.
    
    Args:
        demographics_data: Participant demographics DataFrame
        brain_data: Optional brain imaging data
        performance_data: Optional performance measures
        
    Returns:
        Dict with bias warnings and recommendations
    """
    from .economic_bias import EconomicBiasDetector
    
    detector = EconomicBiasDetector(sensitivity='high', economic_context='japanese')
    
    # Selection bias check
    selection_bias = detector.detect_selection_bias_economic(demographics_data)
    
    warnings = []
    recommendations = []
    
    if selection_bias.get('economic_selection_bias', False):
        warnings.append(f"⚠️  Economic selection bias detected (severity: {selection_bias.get('bias_severity', 0):.2f})")
        recommendations.append("Consider stratified sampling by income/education")
    
    # SES confounding check (if brain/performance data provided)
    if brain_data is not None:
        ses_confounding = detector.detect_ses_confounding(
            demographics_data, brain_data, performance_data
        )
        
        if ses_confounding.get('ses_brain_confounding', False):
            warnings.append(f"⚠️  SES-brain confounding detected (r={ses_confounding.get('ses_brain_correlation', 0):.3f})")
            recommendations.append("Include SES variables as covariates")
    
    # Generate summary
    if warnings:
        summary = "🚨 ECONOMIC BIAS DETECTED - Review recommended"
    else:
        summary = "✅ No major economic bias detected"
    
    return {
        'summary': summary,
        'warnings': warnings,
        'recommendations': recommendations,
        'bias_details': {
            'selection_bias': selection_bias,
            'ses_confounding': ses_confounding if brain_data is not None else None
        }
    }

# Initialize package with economic bias awareness
def _initialize_package():
    """Initialize package with economic bias detection capabilities."""
    import warnings
    
    # Check if user should be warned about economic bias
    warnings.filterwarnings('always', category=UserWarning, module='mcp_fmri')
    
    # Set up logging for economic bias detection
    import logging
    logging.getLogger('mcp_fmri.economic_bias').setLevel(logging.INFO)

# Call initialization
_initialize_package()
