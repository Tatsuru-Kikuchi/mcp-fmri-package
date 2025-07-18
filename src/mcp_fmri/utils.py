#!/usr/bin/env python3
"""
MCP-fMRI Utility Functions
Helper functions for configuration, validation, and ethical guidelines
"""

import yaml
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import logging
import warnings

logger = logging.getLogger(__name__)

def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from YAML or JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                config = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                config = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        logger.info(f"Configuration loaded from {config_path}")
        return config
    
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise

def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Configuration saved to {config_path}")
    
    except Exception as e:
        logger.error(f"Error saving configuration: {e}")
        raise

def validate_data(data: Union[np.ndarray, pd.DataFrame], 
                 data_type: str = "neuroimaging") -> Dict[str, Any]:
    """Validate input data for analysis.
    
    Args:
        data: Input data to validate
        data_type: Type of data being validated
        
    Returns:
        Validation results dictionary
    """
    validation = {
        'valid': True,
        'warnings': [],
        'errors': [],
        'shape': None,
        'data_type': data_type
    }
    
    try:
        if isinstance(data, np.ndarray):
            validation['shape'] = data.shape
            validation['dtype'] = str(data.dtype)
            
            # Check for common issues
            if np.any(np.isnan(data)):
                validation['warnings'].append("Data contains NaN values")
            
            if np.any(np.isinf(data)):
                validation['errors'].append("Data contains infinite values")
                validation['valid'] = False
            
            if data_type == "neuroimaging":
                # Neuroimaging-specific validation
                if len(data.shape) != 2:
                    validation['warnings'].append(f"Expected 2D data (participants x voxels), got {len(data.shape)}D")
                
                if data.shape[0] < 20:
                    validation['warnings'].append(f"Small sample size: {data.shape[0]} participants")
                
                if data.shape[1] < 1000:
                    validation['warnings'].append(f"Low dimensionality: {data.shape[1]} features")
        
        elif isinstance(data, pd.DataFrame):
            validation['shape'] = data.shape
            validation['columns'] = list(data.columns)
            
            # Check for missing data
            missing_data = data.isnull().sum()
            if missing_data.any():
                validation['warnings'].append(f"Missing data in columns: {missing_data[missing_data > 0].to_dict()}")
            
            if data_type == "demographics":
                # Demographics-specific validation
                required_columns = ['participant_id', 'gender']
                missing_required = [col for col in required_columns if col not in data.columns]
                if missing_required:
                    validation['errors'].append(f"Missing required columns: {missing_required}")
                    validation['valid'] = False
                
                # Check gender balance
                if 'gender' in data.columns:
                    gender_counts = data['gender'].value_counts()
                    if len(gender_counts) < 2:
                        validation['warnings'].append("Only one gender category found")
                    else:
                        balance = min(gender_counts) / max(gender_counts)
                        if balance < 0.3:
                            validation['warnings'].append(f"Gender imbalance detected: {balance:.2f}")
        
        else:
            validation['errors'].append(f"Unsupported data type: {type(data)}")
            validation['valid'] = False
    
    except Exception as e:
        validation['errors'].append(f"Validation error: {str(e)}")
        validation['valid'] = False
    
    # Log validation results
    if validation['errors']:
        logger.error(f"Data validation failed: {validation['errors']}")
    elif validation['warnings']:
        logger.warning(f"Data validation warnings: {validation['warnings']}")
    else:
        logger.info("Data validation passed")
    
    return validation

def ethical_guidelines() -> Dict[str, str]:
    """Get ethical guidelines for fMRI gender analysis.
    
    Returns:
        Dictionary of ethical guidelines
    """
    guidelines = {
        'similarity_emphasis': (
            "Prioritize identification of similarities over differences. "
            "Research should focus on commonalities in neural patterns rather than group distinctions."
        ),
        
        'individual_focus': (
            "Emphasize individual variation over group generalizations. "
            "Recognize that individual differences typically exceed group-level patterns."
        ),
        
        'cultural_context': (
            "Integrate appropriate cultural context in all analyses. "
            "Consider how sociocultural factors may influence observed patterns."
        ),
        
        'bias_mitigation': (
            "Implement technical and methodological bias reduction strategies. "
            "Actively detect and mitigate potential sources of bias in data and analysis."
        ),
        
        'transparency': (
            "Ensure all methods, assumptions, and limitations are clearly documented. "
            "Provide sufficient detail for reproducibility and critical evaluation."
        ),
        
        'non_discrimination': (
            "Results should never be used to justify educational or occupational discrimination. "
            "Findings must not reinforce harmful stereotypes or support discriminatory practices."
        ),
        
        'sample_diversity': (
            "Strive for representative and diverse samples. "
            "Consider demographic, regional, and socioeconomic diversity in participant recruitment."
        ),
        
        'effect_size_interpretation': (
            "Interpret effect sizes within appropriate context. "
            "Small effect sizes may be practically meaningless despite statistical significance."
        ),
        
        'replication_emphasis': (
            "Emphasize replication and reproducibility. "
            "Single studies should not be used to make broad generalizations about groups."
        ),
        
        'stakeholder_engagement': (
            "Engage with relevant stakeholders and communities. "
            "Consider the perspectives and concerns of those who may be affected by the research."
        )
    }
    
    return guidelines

def check_ethical_compliance(analysis_config: Dict[str, Any]) -> Dict[str, Any]:
    """Check analysis configuration for ethical compliance.
    
    Args:
        analysis_config: Analysis configuration dictionary
        
    Returns:
        Compliance check results
    """
    compliance = {
        'compliant': True,
        'violations': [],
        'recommendations': [],
        'score': 0
    }
    
    checks = {
        'similarity_focus': analysis_config.get('similarity_focus', False),
        'cultural_context_included': analysis_config.get('cultural_context') is not None,
        'bias_detection_enabled': analysis_config.get('bias_detection', False),
        'individual_emphasis': analysis_config.get('individual_emphasis', False),
        'transparency_documentation': analysis_config.get('documentation_level', 'minimal') in ['detailed', 'comprehensive'],
        'diverse_sampling': analysis_config.get('diverse_sampling', False)
    }
    
    # Evaluate each check
    for check, passed in checks.items():
        if passed:
            compliance['score'] += 1
        else:
            if check == 'similarity_focus':
                compliance['violations'].append("Analysis should focus on similarities rather than differences")
            elif check == 'cultural_context_included':
                compliance['recommendations'].append("Consider including cultural context in analysis")
            elif check == 'bias_detection_enabled':
                compliance['recommendations'].append("Enable bias detection to identify potential issues")
            elif check == 'individual_emphasis':
                compliance['recommendations'].append("Emphasize individual variation over group patterns")
            elif check == 'transparency_documentation':
                compliance['recommendations'].append("Provide detailed documentation for transparency")
            elif check == 'diverse_sampling':
                compliance['recommendations'].append("Ensure diverse and representative sampling")
    
    # Calculate compliance percentage
    compliance['percentage'] = (compliance['score'] / len(checks)) * 100
    
    if compliance['violations']:
        compliance['compliant'] = False
        logger.warning(f"Ethical compliance violations detected: {compliance['violations']}")
    elif compliance['percentage'] < 70:
        logger.warning(f"Low ethical compliance score: {compliance['percentage']:.1f}%")
    else:
        logger.info(f"Ethical compliance check passed: {compliance['percentage']:.1f}%")
    
    return compliance

def generate_sample_config(config_type: str = "preprocessing") -> Dict[str, Any]:
    """Generate sample configuration file.
    
    Args:
        config_type: Type of configuration ('preprocessing' or 'analysis')
        
    Returns:
        Sample configuration dictionary
    """
    if config_type == "preprocessing":
        config = {
            'input_dir': '/path/to/raw/data',
            'output_dir': '/path/to/processed/data',
            'cultural_context': 'japanese',
            'ethical_guidelines': True,
            'motion_threshold': 3.0,
            'rotation_threshold': 3.0,
            'smoothing_fwhm': 8.0,
            'high_pass_cutoff': 128.0,
            'participants': ['JP001', 'JP002', 'JP003'],  # Example list
            'quality_control': {
                'min_snr': 80,
                'min_tsnr': 50,
                'max_motion_volumes': 0.2
            }
        }
    
    elif config_type == "analysis":
        config = {
            'data_dir': '/path/to/processed/data',
            'output_dir': '/path/to/results',
            'cultural_context': 'japanese',
            'similarity_threshold': 0.8,
            'bias_detection': True,
            'similarity_focus': True,
            'individual_emphasis': True,
            'generate_plots': True,
            'interactive_dashboard': True,
            'documentation_level': 'detailed'
        }
    
    else:
        raise ValueError(f"Unknown config type: {config_type}")
    
    return config

def format_results_for_publication(results: Dict[str, Any], 
                                 cultural_context: Optional[str] = None) -> str:
    """Format analysis results for publication.
    
    Args:
        results: Analysis results dictionary
        cultural_context: Cultural context for interpretation
        
    Returns:
        Formatted results string
    """
    output = []
    
    # Header
    output.append("Gender Similarity Analysis Results")
    output.append("=" * 40)
    output.append("")
    
    if cultural_context:
        output.append(f"Cultural Context: {cultural_context}")
        output.append("")
    
    # Key findings
    similarities = results.get('similarities', {})
    
    output.append("Key Findings:")
    output.append(f"• Overall similarity index: {similarities.get('overall_similarity_index', 0):.3f}")
    output.append(f"• Individual to group variation ratio: {similarities.get('individual_to_group_ratio', 0):.2f}:1")
    output.append(f"• Mean effect size (Cohen's d): {similarities.get('mean_cohens_d', 0):.3f}")
    
    # Statistical significance
    if similarities.get('mean_cohens_d', 0) < 0.2:
        output.append("• Effect size classification: Small (supports similarity hypothesis)")
    elif similarities.get('mean_cohens_d', 0) < 0.5:
        output.append("• Effect size classification: Medium")
    else:
        output.append("• Effect size classification: Large")
    
    # Bias detection
    bias_results = results.get('bias_detection', {})
    if bias_results:
        output.append("")
        output.append("Bias Detection:")
        output.append(f"• Classification accuracy: {bias_results.get('classification_accuracy', 0):.3f}")
        output.append(f"• Bias risk level: {bias_results.get('bias_risk', 'unknown')}")
    
    # Ethical considerations
    output.append("")
    output.append("Ethical Considerations:")
    output.append("• Analysis emphasizes similarities over differences")
    output.append("• Individual variation exceeds group-level patterns")
    output.append("• Cultural context integrated in interpretation")
    output.append("• Results should not justify discrimination")
    
    return "\n".join(output)

def create_ethical_checklist() -> List[Dict[str, str]]:
    """Create ethical analysis checklist.
    
    Returns:
        List of checklist items
    """
    checklist = [
        {
            'category': 'Study Design',
            'item': 'Focus on similarities rather than differences',
            'description': 'Research questions should emphasize commonalities'
        },
        {
            'category': 'Study Design',
            'item': 'Include diverse and representative sample',
            'description': 'Ensure demographic, regional, and socioeconomic diversity'
        },
        {
            'category': 'Analysis',
            'item': 'Implement bias detection methods',
            'description': 'Use statistical methods to detect potential bias'
        },
        {
            'category': 'Analysis',
            'item': 'Emphasize individual variation',
            'description': 'Highlight that individual differences exceed group differences'
        },
        {
            'category': 'Interpretation',
            'item': 'Include cultural context',
            'description': 'Consider sociocultural factors in result interpretation'
        },
        {
            'category': 'Interpretation',
            'item': 'Report effect sizes with context',
            'description': 'Interpret practical significance alongside statistical significance'
        },
        {
            'category': 'Reporting',
            'item': 'Use similarity-focused language',
            'description': 'Frame findings in terms of commonalities'
        },
        {
            'category': 'Reporting',
            'item': 'Include ethical disclaimer',
            'description': 'State that results should not justify discrimination'
        },
        {
            'category': 'Dissemination',
            'item': 'Engage with stakeholder communities',
            'description': 'Consider perspectives of affected communities'
        },
        {
            'category': 'Dissemination',
            'item': 'Emphasize limitations and context',
            'description': 'Clearly communicate study limitations and context'
        }
    ]
    
    return checklist