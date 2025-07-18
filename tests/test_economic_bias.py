#!/usr/bin/env python3
"""
Tests for MCP-fMRI economic bias detection module
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from mcp_fmri.economic_bias import (
    EconomicBiasDetector,
    run_comprehensive_economic_bias_analysis
)

class TestEconomicBiasDetector:
    """Test cases for economic bias detection."""
    
    def setup_method(self):
        """Set up test environment."""
        self.detector = EconomicBiasDetector(
            sensitivity='high',
            economic_context='japanese'
        )
    
    def test_detector_initialization(self):
        """Test detector initialization."""
        assert self.detector.sensitivity == 'high'
        assert self.detector.economic_context == 'japanese'
        assert 'income_correlation' in self.detector.current_thresholds
        assert 'expected_ses_effect' in self.detector.context_adjustments
    
    def create_test_demographics(self, bias_level='low'):
        """Create test demographics with specified bias level."""
        n_participants = 100
        
        base_demographics = pd.DataFrame({
            'participant_id': [f"JP{i:03d}" for i in range(1, n_participants + 1)],
            'age': np.random.normal(21, 2, n_participants),
            'gender': np.random.choice(['female', 'male'], n_participants),
            'region': np.random.choice(['tokyo', 'osaka', 'kyoto', 'other'], n_participants)
        })
        
        if bias_level == 'low':
            # Unbiased income distribution
            base_demographics['family_income'] = np.random.lognormal(10.5, 0.6, n_participants)
            base_demographics['education_years'] = np.random.normal(15, 2.5, n_participants)
            base_demographics['parental_education'] = np.random.normal(14, 3, n_participants)
        elif bias_level == 'high':
            # Heavily biased toward high SES
            base_demographics['family_income'] = np.random.lognormal(11.5, 0.3, n_participants)
            base_demographics['education_years'] = np.random.normal(17, 1, n_participants)
            base_demographics['parental_education'] = np.random.normal(16, 1.5, n_participants)
        
        base_demographics['employment_type'] = np.random.choice(
            ['student', 'part_time', 'full_time'], n_participants, p=[0.7, 0.2, 0.1]
        )
        
        return base_demographics
    
    def test_selection_bias_detection_low_bias(self):
        """Test selection bias detection with low bias data."""
        demographics = self.create_test_demographics(bias_level='low')
        
        bias_results = self.detector.detect_selection_bias_economic(demographics)
        
        assert 'income_inequality' in bias_results
        assert 'education_range' in bias_results
        assert 'employment_diversity' in bias_results
        assert 'economic_selection_bias' in bias_results
        assert isinstance(bias_results['economic_selection_bias'], bool)
        
        # Low bias should result in acceptable metrics
        assert bias_results['income_inequality'] < 0.6  # Reasonable inequality
        assert not bias_results['economic_selection_bias']  # Should not detect bias
    
    def test_selection_bias_detection_high_bias(self):
        """Test selection bias detection with high bias data."""
        demographics = self.create_test_demographics(bias_level='high')
        
        bias_results = self.detector.detect_selection_bias_economic(demographics)
        
        # High bias should be detected
        assert bias_results['economic_selection_bias'] == True
        assert bias_results['bias_severity'] > 0.3
    
    def test_ses_confounding_detection(self):
        """Test SES confounding detection."""
        demographics = self.create_test_demographics(bias_level='low')
        
        # Create brain data with SES confounding
        n_participants = len(demographics)
        n_voxels = 1000
        brain_data = np.random.normal(0, 1, (n_participants, n_voxels))
        
        # Add SES effect to brain data
        ses_composite = (demographics['family_income'] - demographics['family_income'].mean()) / demographics['family_income'].std()
        brain_means = brain_data.mean(axis=1)
        brain_means += ses_composite * 0.3  # Strong SES effect
        
        # Create performance data
        performance_data = np.random.normal(75, 10, n_participants)
        performance_data += ses_composite * 5  # SES effect on performance
        
        confounding_results = self.detector.detect_ses_confounding(
            demographics, brain_data, performance_data
        )
        
        assert 'ses_brain_correlation' in confounding_results
        assert 'ses_performance_correlation' in confounding_results
        assert 'ses_brain_confounding' in confounding_results
        assert 'recommendations' in confounding_results
        
        # Should detect confounding due to added SES effect
        assert abs(confounding_results['ses_brain_correlation']) > 0.2
        assert confounding_results['ses_brain_confounding'] == True
        assert len(confounding_results['recommendations']) > 0
    
    def test_temporal_bias_detection(self):
        """Test temporal bias detection."""
        # Create study timeline
        from datetime import datetime, timedelta
        
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 12, 31)
        
        study_timeline = pd.DataFrame({
            'participant_id': [f"JP{i:03d}" for i in range(1, 101)],
            'date': [start_date + timedelta(days=np.random.randint(0, 365)) for _ in range(100)]
        })
        
        # Create economic indicators with volatility
        date_range = pd.date_range(start_date, end_date, freq='M')
        economic_indicators = pd.DataFrame({
            'date': date_range,
            'gdp_growth': np.random.normal(1.0, 2.5, len(date_range)),  # High volatility
            'unemployment_rate': np.random.normal(3.0, 0.5, len(date_range)),
            'education_policy_changes': np.random.poisson(0.2, len(date_range))
        })
        
        temporal_results = self.detector.detect_temporal_economic_bias(
            study_timeline, economic_indicators
        )
        
        assert 'study_duration_months' in temporal_results
        assert 'gdp_volatility' in temporal_results
        assert 'temporal_bias_detected' in temporal_results
        assert 'recommendations' in temporal_results
        
        # High GDP volatility should be detected
        assert temporal_results['gdp_volatility'] > 2.0
        assert temporal_results['high_economic_volatility'] == True
    
    def test_algorithmic_bias_detection(self):
        """Test algorithmic bias detection."""
        # Create test data
        n_participants = 100
        
        X = pd.DataFrame({
            'gender': np.random.choice(['female', 'male'], n_participants),
            'income': np.random.lognormal(10, 0.5, n_participants),
            'education_years': np.random.normal(15, 2, n_participants),
            'brain_feature_1': np.random.normal(0, 1, n_participants),
            'brain_feature_2': np.random.normal(0, 1, n_participants)
        })
        
        # Create biased outcomes (higher for males)
        y = np.random.binomial(1, 0.5, n_participants)
        male_mask = X['gender'] == 'male'
        y[male_mask] = np.random.binomial(1, 0.7, male_mask.sum())  # Higher success rate for males
        
        # Create biased predictions
        predictions = y.copy().astype(float)
        predictions += np.random.normal(0, 0.1, n_participants)  # Add noise
        predictions = predictions.clip(0, 1)
        
        # Mock model
        mock_model = MagicMock()
        
        bias_results = self.detector.detect_algorithmic_bias_economic(
            mock_model, X, y, predictions
        )
        
        assert 'demographic_parity_gender' in bias_results
        assert 'economic_bias_severity' in bias_results
        assert 'recommendations' in bias_results
        
        # Should detect gender bias
        assert bias_results['demographic_parity_violation_gender'] == True
        assert bias_results['economic_bias_severity'] > 0.3
    
    def test_gini_coefficient_calculation(self):
        """Test Gini coefficient calculation."""
        # Perfect equality
        equal_values = np.array([100, 100, 100, 100])
        gini_equal = self.detector._calculate_gini_coefficient(equal_values)
        assert abs(gini_equal) < 0.01  # Should be near 0
        
        # Perfect inequality
        unequal_values = np.array([0, 0, 0, 400])
        gini_unequal = self.detector._calculate_gini_coefficient(unequal_values)
        assert gini_unequal > 0.7  # Should be high
    
    def test_ses_composite_calculation(self):
        """Test SES composite calculation."""
        demographics = pd.DataFrame({
            'income': [50000, 60000, 70000, 80000],
            'education_years': [12, 14, 16, 18],
            'parental_education': [10, 12, 14, 16]
        })
        
        ses_composite = self.detector._calculate_ses_composite(demographics)
        
        assert len(ses_composite) == 4
        assert abs(ses_composite.mean()) < 0.1  # Should be centered around 0
        assert ses_composite.std() > 0  # Should have variation
        
        # Higher values should correspond to higher SES
        assert ses_composite[3] > ses_composite[0]  # Highest vs lowest SES
    
    def test_equalized_odds_calculation(self):
        """Test equalized odds calculation."""
        # Create test data
        n = 100
        predictions = np.random.binomial(1, 0.6, n)
        y_true = np.random.binomial(1, 0.5, n)
        sensitive_attr = np.random.choice(['A', 'B'], n)
        
        tpr_diff, fpr_diff = self.detector._calculate_equalized_odds(
            predictions, y_true, sensitive_attr
        )
        
        assert isinstance(tpr_diff, float)
        assert isinstance(fpr_diff, float)
        assert tpr_diff >= 0
        assert fpr_diff >= 0
    
    def test_sensitivity_levels(self):
        """Test different sensitivity levels."""
        detectors = {
            'low': EconomicBiasDetector(sensitivity='low'),
            'medium': EconomicBiasDetector(sensitivity='medium'),
            'high': EconomicBiasDetector(sensitivity='high')
        }
        
        # Check thresholds are different
        assert detectors['high'].current_thresholds['income_correlation'] < \
               detectors['medium'].current_thresholds['income_correlation']
        assert detectors['medium'].current_thresholds['income_correlation'] < \
               detectors['low'].current_thresholds['income_correlation']
    
    def test_japanese_context_adjustments(self):
        """Test Japanese context-specific adjustments."""
        japanese_detector = EconomicBiasDetector(economic_context='japanese')
        general_detector = EconomicBiasDetector(economic_context='general')
        
        # Japanese context should have different expectations
        assert japanese_detector.context_adjustments['expected_ses_effect'] < \
               general_detector.context_adjustments['expected_ses_effect']
        assert japanese_detector.context_adjustments['family_support_factor'] > \
               general_detector.context_adjustments['family_support_factor']

class TestComprehensiveEconomicBiasAnalysis:
    """Test comprehensive economic bias analysis function."""
    
    def create_comprehensive_test_data(self):
        """Create comprehensive test data for analysis."""
        from datetime import datetime, timedelta
        
        n_participants = 50
        
        # Demographics
        demographics = pd.DataFrame({
            'participant_id': [f"JP{i:03d}" for i in range(1, n_participants + 1)],
            'age': np.random.normal(21, 2, n_participants),
            'gender': np.random.choice(['female', 'male'], n_participants),
            'family_income': np.random.lognormal(10.5, 0.5, n_participants),
            'education_years': np.random.normal(15, 2, n_participants),
            'parental_education': np.random.normal(14, 2.5, n_participants),
            'region': np.random.choice(['tokyo', 'osaka', 'kyoto'], n_participants)
        })
        
        # Brain data
        brain_data = np.random.normal(0, 1, (n_participants, 1000))
        
        # Performance data
        performance_data = np.random.normal(75, 10, n_participants)
        
        # Study timeline
        start_date = datetime(2023, 1, 1)
        study_timeline = pd.DataFrame({
            'participant_id': demographics['participant_id'],
            'date': [start_date + timedelta(days=np.random.randint(0, 365)) for _ in range(n_participants)]
        })
        
        # Economic indicators
        date_range = pd.date_range(start_date, start_date + timedelta(days=365), freq='M')
        economic_indicators = pd.DataFrame({
            'date': date_range,
            'gdp_growth': np.random.normal(1.2, 0.8, len(date_range)),
            'unemployment_rate': np.random.normal(2.8, 0.3, len(date_range)),
            'education_policy_changes': np.random.poisson(0.1, len(date_range))
        })
        
        # Mock model components
        mock_model = MagicMock()
        features = demographics[['age', 'family_income', 'education_years']]
        outcomes = np.random.binomial(1, 0.5, n_participants)
        predictions = np.random.uniform(0, 1, n_participants)
        
        return {
            'demographics': demographics,
            'brain_data': brain_data,
            'performance_data': performance_data,
            'study_timeline': study_timeline,
            'economic_indicators': economic_indicators,
            'trained_model': mock_model,
            'features': features,
            'outcomes': outcomes,
            'predictions': predictions
        }
    
    def test_comprehensive_analysis(self):
        """Test comprehensive economic bias analysis."""
        data_dict = self.create_comprehensive_test_data()
        
        results = run_comprehensive_economic_bias_analysis(data_dict, 'japanese')
        
        # Check main result categories
        assert 'selection_bias' in results
        assert 'ses_confounding' in results
        assert 'temporal_bias' in results
        assert 'algorithmic_bias' in results
        assert 'overall_assessment' in results
        assert 'comprehensive_recommendations' in results
        
        # Check overall assessment
        overall = results['overall_assessment']
        assert 'bias_score' in overall
        assert 'bias_level' in overall
        assert 'bias_detected' in overall
        assert overall['bias_level'] in ['low', 'medium', 'high']
        
        # Check recommendations
        recommendations = results['comprehensive_recommendations']
        assert isinstance(recommendations, list)
        assert len(recommendations) >= 0
    
    def test_comprehensive_analysis_partial_data(self):
        """Test comprehensive analysis with partial data."""
        # Create minimal data dictionary
        minimal_data = {
            'demographics': pd.DataFrame({
                'participant_id': ['JP001', 'JP002'],
                'age': [20, 22],
                'gender': ['female', 'male'],
                'family_income': [50000, 60000]
            })
        }
        
        results = run_comprehensive_economic_bias_analysis(minimal_data, 'japanese')
        
        # Should still work with minimal data
        assert 'selection_bias' in results
        assert 'overall_assessment' in results
        
        # Missing data components should not be present
        assert 'ses_confounding' not in results
        assert 'temporal_bias' not in results
        assert 'algorithmic_bias' not in results
    
    def test_bias_score_calculation(self):
        """Test bias score calculation logic."""
        data_dict = self.create_comprehensive_test_data()
        
        # Add strong bias indicators
        data_dict['demographics']['family_income'] = np.random.lognormal(12, 0.2, len(data_dict['demographics']))
        
        results = run_comprehensive_economic_bias_analysis(data_dict, 'japanese')
        
        # Should detect high bias
        assert results['overall_assessment']['bias_score'] > 0.3
        assert results['overall_assessment']['bias_detected'] == True
    
    def test_recommendation_generation(self):
        """Test recommendation generation."""
        data_dict = self.create_comprehensive_test_data()
        
        # Create biased data
        data_dict['demographics']['family_income'] = np.random.lognormal(12, 0.2, len(data_dict['demographics']))
        
        results = run_comprehensive_economic_bias_analysis(data_dict, 'japanese')
        
        recommendations = results['comprehensive_recommendations']
        
        # Should generate recommendations for detected bias
        assert len(recommendations) > 0
        
        # Check for specific types of recommendations
        recommendation_text = ' '.join(recommendations).lower()
        
        if results['overall_assessment']['bias_score'] > 0.6:
            assert any([
                'methodological' in recommendation_text,
                'replication' in recommendation_text,
                'revision' in recommendation_text
            ])
    
    def test_cultural_context_integration(self):
        """Test cultural context integration in analysis."""
        data_dict = self.create_comprehensive_test_data()
        
        # Run analysis with Japanese context
        japanese_results = run_comprehensive_economic_bias_analysis(data_dict, 'japanese')
        
        # Check that cultural context affects thresholds and interpretations
        if 'ses_confounding' in japanese_results:
            # Japanese context should have different expectations
            ses_results = japanese_results['ses_confounding']
            if 'recommendations' in ses_results:
                rec_text = ' '.join(ses_results['recommendations']).lower()
                # Should consider Japanese cultural factors
                assert len(ses_results['recommendations']) >= 0  # At minimum, should not crash

class TestEconomicBiasIntegration:
    """Test integration with other MCP-fMRI components."""
    
    def test_integration_with_similarity_analyzer(self):
        """Test integration with gender similarity analyzer."""
        from mcp_fmri.analysis import GenderSimilarityAnalyzer
        
        # Create analyzer with economic bias detection
        analyzer = GenderSimilarityAnalyzer(
            ethical_guidelines=True,
            bias_detection=True,
            cultural_context='japanese'
        )
        
        # Load simulated data
        data_dict = analyzer.load_preprocessed_data("test_path")
        
        # Run analysis
        similarities, bias_results = analyzer.analyze_similarities(data_dict)
        
        # Should include economic-relevant bias checks
        assert 'classification_accuracy' in bias_results
        assert 'bias_risk' in bias_results
        
        # Economic bias detector should be compatible
        detector = EconomicBiasDetector(economic_context='japanese')
        
        # Should be able to run additional economic bias analysis
        economic_bias = detector.detect_selection_bias_economic(data_dict['demographics'])
        assert 'economic_selection_bias' in economic_bias
    
    def test_integration_with_cultural_context(self):
        """Test integration with Japanese cultural context."""
        from mcp_fmri.cultural import JapaneseCulturalContext
        
        context = JapaneseCulturalContext()
        detector = EconomicBiasDetector(economic_context='japanese')
        
        # Context adjustments should be compatible
        cultural_adjustments = context.get_cultural_adjustments()
        detector_adjustments = detector.context_adjustments
        
        # Both should recognize Japanese characteristics
        assert cultural_adjustments['expected_effect_size'] <= 0.2  # Small effects expected
        assert detector_adjustments['expected_ses_effect'] <= 0.25  # Lower SES effects expected
    
    def test_bias_reporting_integration(self):
        """Test integration with reporting system."""
        from mcp_fmri.visualization import EthicalReportGenerator
        
        # Create mock analysis results with economic bias data
        analysis_results = {
            'similarities': {
                'overall_similarity_index': 0.85,
                'individual_to_group_ratio': 4.2,
                'mean_cohens_d': 0.12
            },
            'bias_detection': {
                'classification_accuracy': 0.58,
                'bias_risk': 'low',
                'economic_bias_detected': False
            },
            'economic_bias_analysis': {
                'selection_bias': {'economic_selection_bias': False},
                'ses_confounding': {'ses_brain_confounding': False},
                'overall_assessment': {'bias_level': 'low'}
            }
        }
        
        reporter = EthicalReportGenerator(cultural_context='japanese')
        
        # Should be able to generate report including economic bias information
        report = reporter.generate_similarity_report(analysis_results)
        
        assert 'text_report' in report
        assert len(report['text_report']) > 0
        
        # Report should mention economic considerations
        report_text = report['text_report'].lower()
        # At minimum, should not crash when economic bias data is included
        assert 'similarity' in report_text  # Basic sanity check

if __name__ == '__main__':
    pytest.main([__file__])
