#!/usr/bin/env python3
"""
MCP-fMRI Cultural Context Module
Japanese cultural context integration for ethical analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class JapaneseCulturalContext:
    """Japanese cultural context for mathematical cognition research."""
    
    def __init__(
        self,
        education_system: str = "collectivist",
        stereotype_timing: str = "late",
        regional_diversity: bool = True
    ):
        """Initialize Japanese cultural context.
        
        Args:
            education_system: Type of educational approach
            stereotype_timing: When gender stereotypes develop
            regional_diversity: Include regional variations
        """
        self.name = "japanese"
        self.education_system = education_system
        self.stereotype_timing = stereotype_timing
        self.regional_diversity = regional_diversity
        
        # Cultural characteristics based on research
        self.characteristics = {
            'collectivist_culture': True,
            'group_harmony_emphasis': True,
            'late_stereotype_acquisition': True,
            'high_math_competence': True,
            'educational_equality': True,
            'structural_career_barriers': True,
            'family_influence_strong': True,
            'peer_comparison_important': True
        }
        
        # Regional variations
        self.regional_factors = {
            'tokyo': {
                'urbanization': 'high',
                'educational_resources': 'abundant',
                'gender_progressiveness': 'moderate-high'
            },
            'osaka': {
                'urbanization': 'high',
                'educational_resources': 'good',
                'gender_progressiveness': 'moderate'
            },
            'kyoto': {
                'urbanization': 'medium',
                'educational_resources': 'good',
                'gender_progressiveness': 'traditional-moderate'
            },
            'other': {
                'urbanization': 'variable',
                'educational_resources': 'variable',
                'gender_progressiveness': 'traditional'
            }
        }
        
        logger.info(f"Japanese cultural context initialized: {education_system} education system")
    
    def get_cultural_adjustments(self) -> Dict:
        """Get cultural adjustments for analysis.
        
        Returns:
            Dictionary of cultural adjustment factors
        """
        adjustments = {
            # Expect smaller gender differences
            'expected_effect_size': 0.15,  # Cohen's d
            'similarity_expectation': 0.85,  # High similarity expected
            
            # Motion characteristics
            'motion_compliance': 1.2,  # Higher compliance expected
            
            # Performance characteristics
            'math_performance_mean': 76,  # Slightly above global average
            'math_performance_gender_gap': 3,  # Small gap in points
            
            # Stereotype development
            'stereotype_age_onset': 12,  # Later than Western populations
            'stereotype_strength': 0.6,  # Weaker stereotypes
            
            # Educational factors
            'educational_equality': 0.9,  # High equality in education
            'teacher_bias_level': 0.3,  # Lower teacher bias
            
            # Social factors
            'family_support_math': 0.8,  # High family support
            'peer_influence': 0.7,  # Moderate peer influence
            'career_barriers': 0.7  # Structural barriers still exist
        }
        
        return adjustments
    
    def interpret_results(self, similarity_metrics: Dict, bias_metrics: Dict) -> Dict:
        """Interpret results within Japanese cultural context.
        
        Args:
            similarity_metrics: Calculated similarity metrics
            bias_metrics: Bias detection results
            
        Returns:
            Cultural interpretation of results
        """
        adjustments = self.get_cultural_adjustments()
        interpretation = {}
        
        # Similarity interpretation
        expected_similarity = adjustments['similarity_expectation']
        observed_similarity = similarity_metrics.get('overall_similarity_index', 0)
        
        if observed_similarity >= expected_similarity:
            interpretation['similarity_assessment'] = 'consistent_with_japanese_context'
            interpretation['similarity_explanation'] = 'High similarity aligns with Japanese educational equality'
        elif observed_similarity >= 0.7:
            interpretation['similarity_assessment'] = 'moderate_consistency'
            interpretation['similarity_explanation'] = 'Moderate similarity may reflect ongoing cultural changes'
        else:
            interpretation['similarity_assessment'] = 'unexpected_low_similarity'
            interpretation['similarity_explanation'] = 'Lower similarity may indicate sampling or methodological issues'
        
        # Effect size interpretation
        observed_effect = similarity_metrics.get('mean_cohens_d', 0)
        expected_effect = adjustments['expected_effect_size']
        
        if observed_effect <= expected_effect:
            interpretation['effect_size_assessment'] = 'consistent_with_literature'
            interpretation['effect_size_explanation'] = 'Small effect size typical for Japanese populations'
        else:
            interpretation['effect_size_assessment'] = 'larger_than_expected'
            interpretation['effect_size_explanation'] = 'Consider cultural factors or methodological variations'
        
        # Individual variation interpretation
        individual_ratio = similarity_metrics.get('individual_to_group_ratio', 0)
        
        if individual_ratio > 3:
            interpretation['individual_variation_assessment'] = 'individual_differences_dominate'
            interpretation['individual_variation_explanation'] = 'Supports emphasis on individual rather than group patterns'
        else:
            interpretation['individual_variation_assessment'] = 'group_patterns_notable'
            interpretation['individual_variation_explanation'] = 'May reflect cultural group influences'
        
        # Bias assessment in cultural context
        if bias_metrics:
            classification_accuracy = bias_metrics.get('classification_accuracy', 0.5)
            
            if classification_accuracy < 0.6:
                interpretation['bias_assessment'] = 'low_bias_good_similarity'
                interpretation['bias_explanation'] = 'Low classification supports similarity hypothesis'
            elif classification_accuracy < 0.75:
                interpretation['bias_assessment'] = 'moderate_classification'
                interpretation['bias_explanation'] = 'May reflect subtle cultural or methodological factors'
            else:
                interpretation['bias_assessment'] = 'high_classification_concerning'
                interpretation['bias_explanation'] = 'Unexpected high classification - examine methodology'
        
        # Overall cultural conclusion
        interpretation['cultural_conclusion'] = self._generate_cultural_conclusion(
            interpretation, adjustments
        )
        
        return interpretation
    
    def _generate_cultural_conclusion(self, interpretation: Dict, adjustments: Dict) -> str:
        """Generate overall cultural conclusion.
        
        Args:
            interpretation: Analysis interpretation
            adjustments: Cultural adjustments
            
        Returns:
            Cultural conclusion string
        """
        conclusions = []
        
        # Main finding
        if interpretation['similarity_assessment'] == 'consistent_with_japanese_context':
            conclusions.append("Results strongly support gender similarities in Japanese mathematical cognition.")
        else:
            conclusions.append("Results show mixed consistency with expected Japanese patterns.")
        
        # Effect size conclusion
        if interpretation['effect_size_assessment'] == 'consistent_with_literature':
            conclusions.append("Effect sizes align with Japanese research literature.")
        
        # Individual variation
        if interpretation['individual_variation_assessment'] == 'individual_differences_dominate':
            conclusions.append("Individual differences exceed group differences, supporting personalized approaches.")
        
        # Cultural factors
        conclusions.append("Cultural factors including collectivist education and late stereotype acquisition appear influential.")
        
        # Practical implications
        conclusions.append("Findings support evidence-based educational practices emphasizing individual potential over group generalizations.")
        
        return " ".join(conclusions)
    
    def get_demographic_expectations(self, n_participants: int) -> Dict:
        """Get expected demographic characteristics for Japanese samples.
        
        Args:
            n_participants: Number of participants
            
        Returns:
            Expected demographic characteristics
        """
        expectations = {
            'age_range': (18, 25),  # Typical university age
            'age_mean': 20.5,
            'education_years_mean': 14.5,  # Through high school + some university
            'handedness_right_proportion': 0.92,  # Slightly higher than global
            'regional_distribution': {
                'tokyo': 0.35,
                'osaka': 0.20,
                'kyoto': 0.15,
                'other': 0.30
            },
            'math_performance_mean': 76,
            'math_performance_std': 11,
            'gender_balance_target': 0.5  # Aim for equal representation
        }
        
        return expectations
    
    def validate_sample(self, demographics: pd.DataFrame) -> Dict:
        """Validate sample representativeness for Japanese populations.
        
        Args:
            demographics: Sample demographics
            
        Returns:
            Validation results
        """
        expectations = self.get_demographic_expectations(len(demographics))
        validation = {}
        
        # Age validation
        age_mean = demographics['age'].mean()
        expected_age = expectations['age_mean']
        validation['age_appropriate'] = abs(age_mean - expected_age) < 2
        
        # Gender balance
        gender_counts = demographics['gender'].value_counts(normalize=True)
        gender_balance = min(gender_counts) / max(gender_counts)
        validation['gender_balanced'] = gender_balance > 0.4
        
        # Regional representation
        if 'region' in demographics.columns and self.regional_diversity:
            region_counts = demographics['region'].value_counts()
            validation['regionally_diverse'] = len(region_counts) >= 3
        else:
            validation['regionally_diverse'] = True
        
        # Education level
        if 'education_years' in demographics.columns:
            edu_mean = demographics['education_years'].mean()
            expected_edu = expectations['education_years_mean']
            validation['education_appropriate'] = abs(edu_mean - expected_edu) < 2
        else:
            validation['education_appropriate'] = True
        
        # Math performance (if available)
        if 'math_performance' in demographics.columns:
            math_mean = demographics['math_performance'].mean()
            expected_math = expectations['math_performance_mean']
            validation['math_performance_typical'] = abs(math_mean - expected_math) < 10
        else:
            validation['math_performance_typical'] = True
        
        # Overall validation
        validation['sample_representative'] = all([
            validation['age_appropriate'],
            validation['gender_balanced'],
            validation['regionally_diverse'],
            validation['education_appropriate'],
            validation['math_performance_typical']
        ])
        
        if validation['sample_representative']:
            logger.info("Sample appears representative of Japanese university populations")
        else:
            logger.warning("Sample may not be fully representative - consider cultural factors in interpretation")
        
        return validation


class CulturalContextFactory:
    """Factory for creating cultural context objects."""
    
    @staticmethod
    def create_context(culture_name: str, **kwargs) -> object:
        """Create appropriate cultural context object.
        
        Args:
            culture_name: Name of the culture
            **kwargs: Additional parameters
            
        Returns:
            Cultural context object
        """
        if culture_name.lower() == 'japanese':
            return JapaneseCulturalContext(**kwargs)
        else:
            raise ValueError(f"Cultural context '{culture_name}' not implemented")
    
    @staticmethod
    def list_available_contexts() -> List[str]:
        """List available cultural contexts.
        
        Returns:
            List of available culture names
        """
        return ['japanese']