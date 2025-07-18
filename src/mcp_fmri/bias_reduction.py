#!/usr/bin/env python3
"""
MCP-fMRI Bias Reduction Module
Comprehensive bias detection and mitigation with focus on economic applications

This module addresses various types of bias that commonly occur in neuroimaging research
and have significant implications for economic policy and decision-making.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from scipy import stats
import warnings
import logging

logger = logging.getLogger(__name__)

class EconomicBiasDetector:
    """
    Specialized bias detector focusing on economic implications of neuroimaging research.
    
    Economic research is particularly vulnerable to bias because:
    1. Results often inform policy decisions affecting millions
    2. Gender-based findings can perpetuate discrimination in hiring/education
    3. Small effect sizes can be misinterpreted as practically significant
    4. Publication bias favors significant differences over null results
    5. Selection bias in samples can misrepresent population economics
    
    This class implements multiple bias detection strategies used in economics:
    - Selection bias detection (Heckman-style corrections)
    - Publication bias assessment (funnel plot analysis)
    - Statistical power analysis
    - Effect size contextualization
    - Economic significance vs statistical significance
    """
    
    def __init__(self, economic_context: bool = True, policy_implications: bool = True):
        """
        Initialize economic bias detector.
        
        Args:
            economic_context: Enable economic-specific bias checks
            policy_implications: Consider policy implications in bias assessment
        """
        self.economic_context = economic_context
        self.policy_implications = policy_implications
        
        # Economic bias thresholds (more stringent than general research)
        self.thresholds = {
            'classification_accuracy': 0.6,  # Lower threshold for economic policy
            'effect_size_economic_significance': 0.1,  # Practical significance threshold
            'sample_representativeness': 0.8,  # Higher standard for policy research
            'publication_bias_p_value': 0.05,  # Standard p-hacking threshold
            'statistical_power': 0.8  # Minimum power for reliable conclusions
        }
        
        logger.info("Economic bias detector initialized with policy-aware thresholds")
    
    def detect_selection_bias(self, demographics: pd.DataFrame, 
                            participation_data: Optional[pd.DataFrame] = None) -> Dict:
        """
        Detect selection bias that could skew economic interpretations.
        
        Selection bias in economics occurs when:
        - High-achieving individuals are overrepresented (survivorship bias)
        - Certain socioeconomic groups are systematically excluded
        - Geographic clustering creates non-representative samples
        - Self-selection creates motivation confounds
        
        This is critical for economic research because policy decisions based on
        biased samples can exacerbate inequality rather than address it.
        
        Args:
            demographics: Sample demographics
            participation_data: Optional data on participation rates by group
            
        Returns:
            Dictionary with selection bias indicators
        """
        logger.info("Analyzing selection bias with economic implications")
        
        bias_indicators = {
            'selection_bias_detected': False,
            'bias_sources': [],
            'economic_risk_level': 'low',
            'policy_reliability': 'high'
        }
        
        # 1. Socioeconomic representativeness
        if 'education_years' in demographics.columns:
            edu_mean = demographics['education_years'].mean()
            edu_std = demographics['education_years'].std()
            
            # Check for over-representation of highly educated
            if edu_mean > 16:  # Above typical population mean
                bias_indicators['bias_sources'].append('highly_educated_overrepresented')
                bias_indicators['selection_bias_detected'] = True
                
            # Check for insufficient educational diversity
            if edu_std < 2:  # Low diversity
                bias_indicators['bias_sources'].append('insufficient_educational_diversity')
                bias_indicators['selection_bias_detected'] = True
        
        # 2. Geographic clustering bias
        if 'region' in demographics.columns:
            region_counts = demographics['region'].value_counts(normalize=True)
            max_region_prop = region_counts.max()
            
            if max_region_prop > 0.6:  # One region dominates
                bias_indicators['bias_sources'].append('geographic_clustering')
                bias_indicators['selection_bias_detected'] = True
                
                # Urban bias is particularly problematic for economic policy
                if region_counts.index[0] in ['tokyo', 'urban']:
                    bias_indicators['bias_sources'].append('urban_bias')
        
        # 3. Age-related selection bias
        if 'age' in demographics.columns:
            age_range = demographics['age'].max() - demographics['age'].min()
            age_mean = demographics['age'].mean()
            
            # Check for narrow age range (common in university samples)
            if age_range < 5:
                bias_indicators['bias_sources'].append('narrow_age_range')
                bias_indicators['selection_bias_detected'] = True
            
            # Check for young adult bias
            if age_mean < 25:
                bias_indicators['bias_sources'].append('young_adult_bias')
                bias_indicators['selection_bias_detected'] = True
        
        # 4. Gender balance (critical for economic policy)
        if 'gender' in demographics.columns:
            gender_counts = demographics['gender'].value_counts(normalize=True)
            gender_balance = min(gender_counts) / max(gender_counts)
            
            if gender_balance < 0.4:  # Significant imbalance
                bias_indicators['bias_sources'].append('gender_imbalance')
                bias_indicators['selection_bias_detected'] = True
        
        # 5. Participation rate bias (if data available)
        if participation_data is not None:
            # Analyze differential participation rates by group
            if 'participation_rate' in participation_data.columns:
                participation_by_group = participation_data.groupby('group')['participation_rate'].mean()
                participation_variance = participation_by_group.var()
                
                if participation_variance > 0.05:  # High variance in participation
                    bias_indicators['bias_sources'].append('differential_participation_rates')
                    bias_indicators['selection_bias_detected'] = True
        
        # 6. Economic risk assessment
        num_bias_sources = len(bias_indicators['bias_sources'])
        
        if num_bias_sources >= 3:
            bias_indicators['economic_risk_level'] = 'high'
            bias_indicators['policy_reliability'] = 'low'
        elif num_bias_sources >= 2:
            bias_indicators['economic_risk_level'] = 'medium'
            bias_indicators['policy_reliability'] = 'medium'
        
        # 7. Economic implications
        economic_implications = self._assess_economic_implications(bias_indicators)
        bias_indicators.update(economic_implications)
        
        return bias_indicators
    
    def detect_publication_bias(self, effect_sizes: List[float], 
                              standard_errors: List[float],
                              study_metadata: Optional[pd.DataFrame] = None) -> Dict:
        """
        Detect publication bias using economic research standards.
        
        Publication bias is particularly problematic in economic-relevant research because:
        - Studies showing gender differences are more likely to be published
        - Null results (similarity findings) are systematically underreported
        - This creates false impression of differences in economic capability
        - Policy makers may base decisions on biased literature
        
        Uses multiple detection methods:
        1. Funnel plot asymmetry test
        2. Egger's test for small-study effects
        3. p-curve analysis
        4. Economic significance vs statistical significance
        
        Args:
            effect_sizes: List of observed effect sizes
            standard_errors: Corresponding standard errors
            study_metadata: Optional metadata about studies
            
        Returns:
            Publication bias assessment
        """
        logger.info("Analyzing publication bias with economic policy implications")
        
        bias_assessment = {
            'publication_bias_detected': False,
            'bias_strength': 'none',
            'economic_reliability': 'high',
            'policy_recommendations': []
        }
        
        effect_sizes = np.array(effect_sizes)
        standard_errors = np.array(standard_errors)
        
        # 1. Funnel plot asymmetry (Egger's test)
        if len(effect_sizes) >= 10:  # Need sufficient studies
            precision = 1 / standard_errors
            
            # Regression of effect size on precision
            slope, intercept, r_value, p_value, std_err = stats.linregress(precision, effect_sizes)
            
            # Significant intercept suggests publication bias
            if abs(intercept) > 0.1 and p_value < 0.05:
                bias_assessment['publication_bias_detected'] = True
                bias_assessment['bias_strength'] = 'moderate' if abs(intercept) < 0.2 else 'strong'
                bias_assessment['eggers_test_p'] = p_value
                bias_assessment['eggers_intercept'] = intercept
        
        # 2. Small-study effects
        if len(effect_sizes) >= 5:
            # Correlation between effect size and standard error
            correlation, p_val = stats.pearsonr(effect_sizes, standard_errors)
            
            if correlation > 0.3 and p_val < 0.05:
                bias_assessment['small_study_effects'] = True
                bias_assessment['publication_bias_detected'] = True
        
        # 3. Economic significance assessment
        economically_significant = np.abs(effect_sizes) > self.thresholds['effect_size_economic_significance']
        prop_economically_significant = np.mean(economically_significant)
        
        # If too many studies show "economic significance", suspect bias
        if prop_economically_significant > 0.7:
            bias_assessment['excess_economic_significance'] = True
            bias_assessment['publication_bias_detected'] = True
        
        # 4. P-value distribution analysis
        if study_metadata is not None and 'p_value' in study_metadata.columns:
            p_values = study_metadata['p_value'].dropna()
            
            # Check for p-hacking (excess of p-values just below 0.05)
            p_vals_near_threshold = p_values[(p_values >= 0.04) & (p_values <= 0.05)]
            prop_near_threshold = len(p_vals_near_threshold) / len(p_values)
            
            if prop_near_threshold > 0.2:  # More than 20% near threshold
                bias_assessment['p_hacking_suspected'] = True
                bias_assessment['publication_bias_detected'] = True
        
        # 5. Economic policy implications
        if bias_assessment['publication_bias_detected']:
            bias_assessment['economic_reliability'] = 'low'
            bias_assessment['policy_recommendations'].extend([
                'Require pre-registration of neuroimaging studies',
                'Mandate reporting of null results in policy-relevant research',
                'Use meta-analytic approaches for policy decisions',
                'Consider economic significance alongside statistical significance'
            ])
        
        return bias_assessment
    
    def assess_statistical_power(self, sample_size: int, effect_size: float, 
                               alpha: float = 0.05, test_type: str = 'two_sample') -> Dict:
        """
        Assess statistical power with economic research standards.
        
        Inadequate statistical power is a major issue in economic research because:
        - Underpowered studies produce unreliable effect estimates
        - Policy decisions based on underpowered research can be harmful
        - Type II errors (missing true effects) can perpetuate inequality
        - Economic research often requires higher power for practical significance
        
        Args:
            sample_size: Total sample size
            effect_size: Expected or observed effect size
            alpha: Significance level
            test_type: Type of statistical test
            
        Returns:
            Power analysis results with economic implications
        """
        logger.info(f"Assessing statistical power for economic research (n={sample_size}, d={effect_size})")
        
        # Simplified power calculation for two-sample t-test
        if test_type == 'two_sample':
            # Cohen's formula for power
            n_per_group = sample_size / 2
            delta = effect_size * np.sqrt(n_per_group / 2)
            
            # Approximate power using normal distribution
            z_alpha = stats.norm.ppf(1 - alpha/2)
            z_beta = delta - z_alpha
            power = stats.norm.cdf(z_beta)
        else:
            # Default approximation
            power = 1 - stats.norm.cdf(stats.norm.ppf(1 - alpha/2) - effect_size * np.sqrt(sample_size/4))
        
        power_assessment = {
            'statistical_power': power,
            'power_adequate': power >= self.thresholds['statistical_power'],
            'economic_reliability': 'high' if power >= 0.8 else 'medium' if power >= 0.6 else 'low',
            'sample_size': sample_size,
            'effect_size': effect_size
        }
        
        # Economic-specific power recommendations
        if power < 0.8:
            required_n = self._calculate_required_sample_size(effect_size, alpha, 0.8)
            power_assessment['required_sample_size'] = required_n
            power_assessment['economic_recommendations'] = [
                f'Increase sample size to {required_n} for adequate power',
                'Consider multi-site collaboration for larger samples',
                'Use Bayesian methods to incorporate prior information',
                'Focus on confidence intervals rather than p-values'
            ]
        
        # Policy implications
        if power < 0.6:
            power_assessment['policy_implications'] = [
                'Results insufficient for policy decisions',
                'Require replication before policy implementation',
                'Consider as preliminary evidence only'
            ]
        
        return power_assessment
    
    def economic_significance_test(self, effect_size: float, confidence_interval: Tuple[float, float],
                                 sample_size: int, policy_context: str = 'education') -> Dict:
        """
        Test for economic significance beyond statistical significance.
        
        Economic significance considers:
        1. Practical importance of the effect size
        2. Cost-benefit analysis of policy interventions
        3. Real-world impact on individuals and society
        4. Implementation feasibility
        
        This is crucial because statistically significant findings may not be
        economically meaningful, and vice versa.
        
        Args:
            effect_size: Observed effect size (Cohen's d)
            confidence_interval: 95% confidence interval for effect size
            sample_size: Sample size
            policy_context: Context for economic significance ('education', 'employment', 'healthcare')
            
        Returns:
            Economic significance assessment
        """
        logger.info(f"Assessing economic significance in {policy_context} context")
        
        # Policy-specific thresholds for economic significance
        economic_thresholds = {
            'education': {
                'minimal': 0.05,    # Barely detectable in classroom
                'small': 0.1,       # Noticeable to teachers
                'moderate': 0.2,    # Affects educational outcomes
                'large': 0.4        # Major policy consideration
            },
            'employment': {
                'minimal': 0.1,     # Small hiring bias
                'small': 0.2,       # Noticeable workplace differences
                'moderate': 0.3,    # Significant career impact
                'large': 0.5        # Major employment discrimination
            },
            'healthcare': {
                'minimal': 0.05,    # Clinical relevance threshold
                'small': 0.1,       # Treatment consideration
                'moderate': 0.25,   # Clinical significance
                'large': 0.4        # Major health implications
            }
        }
        
        thresholds = economic_thresholds.get(policy_context, economic_thresholds['education'])
        
        abs_effect = abs(effect_size)
        ci_lower, ci_upper = confidence_interval
        
        # Determine economic significance level
        if abs_effect >= thresholds['large']:
            significance_level = 'large'
        elif abs_effect >= thresholds['moderate']:
            significance_level = 'moderate'
        elif abs_effect >= thresholds['small']:
            significance_level = 'small'
        elif abs_effect >= thresholds['minimal']:
            significance_level = 'minimal'
        else:
            significance_level = 'negligible'
        
        # Check confidence interval for practical significance
        ci_contains_zero = ci_lower <= 0 <= ci_upper
        ci_entirely_negligible = abs(ci_upper) < thresholds['minimal'] and abs(ci_lower) < thresholds['minimal']
        
        assessment = {
            'effect_size': effect_size,
            'economic_significance_level': significance_level,
            'economically_significant': significance_level in ['small', 'moderate', 'large'],
            'confidence_interval': confidence_interval,
            'ci_contains_zero': ci_contains_zero,
            'ci_entirely_negligible': ci_entirely_negligible,
            'policy_context': policy_context
        }
        
        # Economic interpretation
        if significance_level == 'negligible' or ci_entirely_negligible:
            assessment['economic_interpretation'] = 'No meaningful economic impact'
            assessment['policy_recommendation'] = 'Focus on individual assessment rather than group differences'
        elif significance_level == 'minimal':
            assessment['economic_interpretation'] = 'Minimal economic impact, unlikely to justify policy changes'
            assessment['policy_recommendation'] = 'Monitor but do not base policy decisions on this finding'
        elif significance_level == 'small':
            assessment['economic_interpretation'] = 'Small economic impact, consider broader context'
            assessment['policy_recommendation'] = 'Consider as one factor among many in policy decisions'
        elif significance_level == 'moderate':
            assessment['economic_interpretation'] = 'Moderate economic impact, warrants policy consideration'
            assessment['policy_recommendation'] = 'Important factor in policy decisions, but consider individual variation'
        else:  # large
            assessment['economic_interpretation'] = 'Large economic impact, major policy implications'
            assessment['policy_recommendation'] = 'Significant policy consideration, but verify through replication'
        
        # Cost-benefit considerations
        if self.policy_implications:
            cost_benefit = self._estimate_policy_costs(effect_size, sample_size, policy_context)
            assessment.update(cost_benefit)
        
        return assessment
    
    def _assess_economic_implications(self, bias_indicators: Dict) -> Dict:
        """
        Assess economic implications of detected biases.
        
        Args:
            bias_indicators: Bias detection results
            
        Returns:
            Economic implications
        """
        implications = {
            'labor_market_implications': [],
            'educational_policy_implications': [],
            'research_funding_implications': [],
            'social_welfare_implications': []
        }
        
        bias_sources = bias_indicators.get('bias_sources', [])
        
        # Labor market implications
        if 'highly_educated_overrepresented' in bias_sources:
            implications['labor_market_implications'].append(
                'Results may not apply to general workforce'
            )
        
        if 'gender_imbalance' in bias_sources:
            implications['labor_market_implications'].append(
                'Hiring decisions based on biased sample could perpetuate discrimination'
            )
        
        # Educational policy implications
        if 'narrow_age_range' in bias_sources:
            implications['educational_policy_implications'].append(
                'Findings may not apply across all educational levels'
            )
        
        if 'urban_bias' in bias_sources:
            implications['educational_policy_implications'].append(
                'Rural educational policies may be inadequately informed'
            )
        
        # Research funding implications
        if len(bias_sources) >= 2:
            implications['research_funding_implications'].append(
                'Future research should prioritize representative sampling'
            )
            implications['research_funding_implications'].append(
                'Multi-site studies needed to address geographic bias'
            )
        
        # Social welfare implications
        if bias_indicators.get('economic_risk_level') == 'high':
            implications['social_welfare_implications'].append(
                'High risk of policy decisions that could increase inequality'
            )
            implications['social_welfare_implications'].append(
                'Recommend additional validation studies before policy implementation'
            )
        
        return implications
    
    def _calculate_required_sample_size(self, effect_size: float, alpha: float, power: float) -> int:
        """
        Calculate required sample size for desired power.
        
        Args:
            effect_size: Expected effect size
            alpha: Significance level
            power: Desired statistical power
            
        Returns:
            Required sample size
        """
        # Simplified calculation for two-sample t-test
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_beta = stats.norm.ppf(power)
        
        n_per_group = 2 * ((z_alpha + z_beta) / effect_size)**2
        total_n = int(np.ceil(2 * n_per_group))
        
        return total_n
    
    def _estimate_policy_costs(self, effect_size: float, sample_size: int, policy_context: str) -> Dict:
        """
        Estimate economic costs of policy decisions based on effect size.
        
        Args:
            effect_size: Observed effect size
            sample_size: Sample size
            policy_context: Policy area
            
        Returns:
            Cost-benefit estimates
        """
        # Simplified cost-benefit framework
        cost_estimates = {
            'implementation_cost': 'unknown',
            'opportunity_cost': 'unknown',
            'potential_harm_cost': 'unknown',
            'cost_benefit_ratio': 'requires_further_analysis'
        }
        
        abs_effect = abs(effect_size)
        
        if policy_context == 'education':
            if abs_effect < 0.1:
                cost_estimates['implementation_cost'] = 'low_but_questionable_benefit'
            elif abs_effect < 0.2:
                cost_estimates['implementation_cost'] = 'moderate_with_uncertain_benefit'
            else:
                cost_estimates['implementation_cost'] = 'high_but_potentially_justified'
        
        # Potential harm assessment
        if abs_effect > 0.3:
            cost_estimates['potential_harm_cost'] = 'high_if_findings_are_wrong'
            cost_estimates['recommendation'] = 'require_replication_before_policy_change'
        
        return cost_estimates


class BiasReductionPipeline:
    """
    Comprehensive bias reduction pipeline for economic research applications.
    
    This pipeline implements multiple bias reduction strategies:
    1. Pre-analysis bias prevention
    2. Analysis-stage bias detection
    3. Post-analysis bias correction
    4. Economic significance assessment
    5. Policy recommendation framework
    """
    
    def __init__(self, economic_focus: bool = True):
        """
        Initialize bias reduction pipeline.
        
        Args:
            economic_focus: Enable economic-specific bias reduction
        """
        self.economic_focus = economic_focus
        self.bias_detector = EconomicBiasDetector()
        self.bias_log = []
        
        logger.info("Bias reduction pipeline initialized for economic research")
    
    def run_comprehensive_bias_assessment(self, data: Dict, metadata: Optional[Dict] = None) -> Dict:
        """
        Run comprehensive bias assessment for economic research.
        
        Args:
            data: Analysis data including brain_data and demographics
            metadata: Optional metadata about study design
            
        Returns:
            Comprehensive bias assessment
        """
        logger.info("Running comprehensive bias assessment for economic applications")
        
        assessment = {
            'selection_bias': {},
            'statistical_power': {},
            'economic_significance': {},
            'overall_risk_level': 'unknown',
            'policy_recommendations': [],
            'bias_mitigation_strategies': []
        }
        
        demographics = data.get('demographics')
        brain_data = data.get('brain_data')
        
        if demographics is not None:
            # 1. Selection bias assessment
            selection_bias = self.bias_detector.detect_selection_bias(demographics)
            assessment['selection_bias'] = selection_bias
            
            # 2. Statistical power assessment
            sample_size = len(demographics)
            # Estimate effect size from data or use typical values
            estimated_effect_size = metadata.get('expected_effect_size', 0.2) if metadata else 0.2
            
            power_assessment = self.bias_detector.assess_statistical_power(
                sample_size, estimated_effect_size
            )
            assessment['statistical_power'] = power_assessment
            
            # 3. Economic significance assessment
            if 'effect_size' in metadata or brain_data is not None:
                if brain_data is not None:
                    # Calculate effect size from actual data
                    effect_size = self._calculate_effect_size(brain_data, demographics)
                    ci = self._calculate_confidence_interval(brain_data, demographics)
                else:
                    effect_size = metadata.get('effect_size', 0.0)
                    ci = metadata.get('confidence_interval', (-0.1, 0.1))
                
                economic_sig = self.bias_detector.economic_significance_test(
                    effect_size, ci, sample_size
                )
                assessment['economic_significance'] = economic_sig
        
        # 4. Overall risk assessment
        risk_factors = []
        
        if assessment['selection_bias'].get('selection_bias_detected', False):
            risk_factors.append('selection_bias')
        
        if not assessment['statistical_power'].get('power_adequate', True):
            risk_factors.append('insufficient_power')
        
        if assessment['economic_significance'].get('ci_contains_zero', True):
            risk_factors.append('uncertain_economic_significance')
        
        # Determine overall risk level
        if len(risk_factors) >= 3:
            assessment['overall_risk_level'] = 'high'
        elif len(risk_factors) >= 2:
            assessment['overall_risk_level'] = 'medium'
        elif len(risk_factors) >= 1:
            assessment['overall_risk_level'] = 'low'
        else:
            assessment['overall_risk_level'] = 'minimal'
        
        # 5. Generate recommendations
        recommendations = self._generate_bias_reduction_recommendations(assessment)
        assessment['policy_recommendations'] = recommendations['policy']
        assessment['bias_mitigation_strategies'] = recommendations['mitigation']
        
        # Log assessment
        self.bias_log.append(assessment)
        
        return assessment
    
    def _calculate_effect_size(self, brain_data: np.ndarray, demographics: pd.DataFrame) -> float:
        """
        Calculate Cohen's d effect size from brain data.
        
        Args:
            brain_data: Brain imaging data
            demographics: Demographic information
            
        Returns:
            Effect size (Cohen's d)
        """
        if 'gender' not in demographics.columns:
            return 0.0
        
        female_mask = demographics['gender'] == 'female'
        male_mask = demographics['gender'] == 'male'
        
        if not female_mask.any() or not male_mask.any():
            return 0.0
        
        female_data = brain_data[female_mask]
        male_data = brain_data[male_mask]
        
        # Calculate Cohen's d for each voxel
        female_mean = np.mean(female_data, axis=0)
        male_mean = np.mean(male_data, axis=0)
        
        pooled_std = np.sqrt((np.var(female_data, axis=0) + np.var(male_data, axis=0)) / 2)
        cohens_d = (female_mean - male_mean) / pooled_std
        
        # Return mean effect size across all voxels
        return np.mean(np.abs(cohens_d))
    
    def _calculate_confidence_interval(self, brain_data: np.ndarray, 
                                     demographics: pd.DataFrame) -> Tuple[float, float]:
        """
        Calculate confidence interval for effect size.
        
        Args:
            brain_data: Brain imaging data
            demographics: Demographic information
            
        Returns:
            95% confidence interval for effect size
        """
        effect_size = self._calculate_effect_size(brain_data, demographics)
        
        # Simplified CI calculation (should use proper bootstrap in practice)
        n = len(demographics)
        se = np.sqrt(4 / n)  # Approximate standard error for Cohen's d
        
        ci_lower = effect_size - 1.96 * se
        ci_upper = effect_size + 1.96 * se
        
        return (ci_lower, ci_upper)
    
    def _generate_bias_reduction_recommendations(self, assessment: Dict) -> Dict:
        """
        Generate specific recommendations for bias reduction.
        
        Args:
            assessment: Bias assessment results
            
        Returns:
            Recommendations for policy and bias mitigation
        """
        recommendations = {
            'policy': [],
            'mitigation': []
        }
        
        risk_level = assessment['overall_risk_level']
        
        # Risk-level specific recommendations
        if risk_level == 'high':
            recommendations['policy'].extend([
                'DO NOT use findings for immediate policy decisions',
                'Require replication in independent samples',
                'Conduct systematic review of existing literature',
                'Consider meta-analytic approaches'
            ])
            
            recommendations['mitigation'].extend([
                'Increase sample size significantly',
                'Implement stratified sampling for representativeness',
                'Use multiple recruitment sites',
                'Apply statistical corrections for selection bias'
            ])
        
        elif risk_level == 'medium':
            recommendations['policy'].extend([
                'Use findings as preliminary evidence only',
                'Combine with other research before policy decisions',
                'Monitor implementation carefully if policies are based on this research'
            ])
            
            recommendations['mitigation'].extend([
                'Address identified bias sources',
                'Improve sample representativeness',
                'Use robust statistical methods',
                'Report confidence intervals alongside point estimates'
            ])
        
        elif risk_level == 'low':
            recommendations['policy'].extend([
                'Findings can inform policy but consider broader context',
                'Emphasize individual assessment over group generalizations',
                'Monitor for unintended consequences of policy implementation'
            ])
            
            recommendations['mitigation'].extend([
                'Continue monitoring for bias in future studies',
                'Maintain transparency in reporting',
                'Consider cultural context in interpretation'
            ])
        
        # Specific bias-based recommendations
        if assessment['selection_bias'].get('selection_bias_detected', False):
            recommendations['mitigation'].extend([
                'Implement probability-based sampling',
                'Use demographic weighting',
                'Address identified selection mechanisms'
            ])
        
        if not assessment['statistical_power'].get('power_adequate', True):
            recommendations['mitigation'].extend([
                'Increase sample size for adequate power',
                'Use Bayesian methods to incorporate prior information',
                'Focus on effect size estimation rather than hypothesis testing'
            ])
        
        return recommendations


def demonstrate_economic_bias_detection():
    """
    Demonstration of economic bias detection capabilities.
    
    This function shows how the bias detection system works with
    realistic economic research scenarios.
    """
    print("🏦 ECONOMIC BIAS DETECTION DEMONSTRATION")
    print("=" * 45)
    
    # Create example data with various biases
    np.random.seed(42)
    
    # Scenario 1: Highly educated, urban-biased sample
    biased_demographics = pd.DataFrame({
        'participant_id': [f'P{i:03d}' for i in range(1, 81)],
        'gender': ['female'] * 40 + ['male'] * 40,
        'age': np.random.normal(22, 1.5, 80),  # Narrow age range
        'education_years': np.random.normal(17, 1, 80),  # Highly educated
        'region': ['tokyo'] * 60 + ['osaka'] * 20,  # Urban bias
        'income_level': np.random.normal(60000, 10000, 80)  # High income
    })
    
    print("\n📊 SCENARIO 1: Biased Sample Analysis")
    print("-" * 35)
    
    detector = EconomicBiasDetector()
    selection_bias = detector.detect_selection_bias(biased_demographics)
    
    print(f"Selection bias detected: {selection_bias['selection_bias_detected']}")
    print(f"Economic risk level: {selection_bias['economic_risk_level']}")
    print(f"Policy reliability: {selection_bias['policy_reliability']}")
    print(f"Bias sources: {selection_bias['bias_sources']}")
    
    # Scenario 2: Publication bias analysis
    print("\n📈 SCENARIO 2: Publication Bias Analysis")
    print("-" * 38)
    
    # Simulate effect sizes with publication bias (favoring significant results)
    effect_sizes = []
    standard_errors = []
    
    for i in range(20):
        true_effect = np.random.normal(0.05, 0.1)  # Small true effect
        se = np.random.uniform(0.1, 0.3)
        
        # Publication bias: only "publish" if p < 0.05 or effect > 0.2
        z_score = true_effect / se
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        if p_value < 0.05 or abs(true_effect) > 0.2:
            effect_sizes.append(true_effect)
            standard_errors.append(se)
    
    pub_bias = detector.detect_publication_bias(effect_sizes, standard_errors)
    
    print(f"Publication bias detected: {pub_bias['publication_bias_detected']}")
    print(f"Economic reliability: {pub_bias['economic_reliability']}")
    print(f"Bias strength: {pub_bias['bias_strength']}")
    
    # Scenario 3: Economic significance assessment
    print("\n💰 SCENARIO 3: Economic Significance Assessment")
    print("-" * 45)
    
    effect_size = 0.15
    confidence_interval = (0.05, 0.25)
    sample_size = 100
    
    econ_sig = detector.economic_significance_test(
        effect_size, confidence_interval, sample_size, 'education'
    )
    
    print(f"Effect size: {econ_sig['effect_size']:.3f}")
    print(f"Economic significance: {econ_sig['economic_significance_level']}")
    print(f"Economically significant: {econ_sig['economically_significant']}")
    print(f"Policy recommendation: {econ_sig['policy_recommendation']}")
    
    # Scenario 4: Statistical power analysis
    print("\n⚡ SCENARIO 4: Statistical Power Analysis")
    print("-" * 37)
    
    power_analysis = detector.assess_statistical_power(80, 0.2)
    
    print(f"Statistical power: {power_analysis['statistical_power']:.3f}")
    print(f"Power adequate: {power_analysis['power_adequate']}")
    print(f"Economic reliability: {power_analysis['economic_reliability']}")
    
    if 'required_sample_size' in power_analysis:
        print(f"Required sample size: {power_analysis['required_sample_size']}")
    
    print("\n🎯 KEY TAKEAWAYS FOR ECONOMIC RESEARCH:")
    print("=" * 40)
    print("1. Selection bias can severely limit policy applicability")
    print("2. Publication bias creates false impressions of gender differences")
    print("3. Statistical significance ≠ economic significance")
    print("4. Adequate statistical power is crucial for reliable policy decisions")
    print("5. Bias detection should be mandatory for policy-relevant research")


if __name__ == "__main__":
    demonstrate_economic_bias_detection()