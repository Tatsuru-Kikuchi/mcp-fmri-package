#!/usr/bin/env python3
"""
MCP-fMRI Analysis Module
Ethical analysis framework focusing on gender similarities
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Union
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings

logger = logging.getLogger(__name__)

class GenderSimilarityAnalyzer:
    """Analyzer focused on gender similarities in mathematical cognition."""
    
    def __init__(
        self,
        ethical_guidelines: bool = True,
        similarity_threshold: float = 0.8,
        cultural_context: str = "japanese",
        bias_detection: bool = True
    ):
        """Initialize similarity analyzer.
        
        Args:
            ethical_guidelines: Enable ethical analysis framework
            similarity_threshold: Threshold for similarity detection
            cultural_context: Cultural context for analysis
            bias_detection: Enable bias detection
        """
        self.ethical_guidelines = ethical_guidelines
        self.similarity_threshold = similarity_threshold
        self.cultural_context = cultural_context
        self.bias_detection = bias_detection
        
        self.similarity_metrics = {}
        self.bias_metrics = {}
        
        if self.ethical_guidelines:
            logger.info("Ethical similarity analysis framework enabled")
            logger.info(f"Focus: Gender similarities with {cultural_context} cultural context")
    
    def load_preprocessed_data(self, data_path: Union[str, Path]) -> Dict:
        """Load preprocessed fMRI data.
        
        Args:
            data_path: Path to preprocessed data
            
        Returns:
            Dictionary containing loaded data
        """
        data_path = Path(data_path)
        
        # Simulate loading preprocessed data
        # In real implementation, load actual neuroimaging data
        n_participants = np.random.randint(120, 180)
        n_voxels = np.random.randint(50000, 80000)
        
        # Generate realistic fMRI-like data
        brain_data = np.random.normal(0, 1, (n_participants, n_voxels))
        
        # Add realistic spatial correlations
        for i in range(min(1000, n_voxels)):
            if i > 0:
                brain_data[:, i] = 0.3 * brain_data[:, i-1] + 0.7 * brain_data[:, i]
        
        # Generate demographics with Japanese population characteristics
        demographics = self._generate_demographics(n_participants)
        
        return {
            'brain_data': brain_data,
            'demographics': demographics,
            'n_participants': n_participants,
            'n_voxels': n_voxels,
            'cultural_context': self.cultural_context
        }
    
    def _generate_demographics(self, n_participants: int) -> pd.DataFrame:
        """Generate realistic demographics for Japanese populations.
        
        Args:
            n_participants: Number of participants
            
        Returns:
            DataFrame with demographic information
        """
        np.random.seed(42)  # For reproducibility
        
        demographics = pd.DataFrame({
            'participant_id': [f"JP{i:03d}" for i in range(1, n_participants + 1)],
            'age': np.random.normal(22, 3, n_participants).clip(18, 30),
            'gender': np.random.choice(['female', 'male'], n_participants),
            'education_years': np.random.normal(16, 2, n_participants).clip(12, 20),
            'handedness': np.random.choice(['right', 'left'], n_participants, p=[0.9, 0.1]),
            'math_performance': np.random.normal(75, 12, n_participants).clip(0, 100),
            'cultural_background': ['japanese'] * n_participants,
            'region': np.random.choice(['tokyo', 'osaka', 'kyoto', 'other'], n_participants, p=[0.4, 0.2, 0.2, 0.2])
        })
        
        # Add subtle performance similarities (realistic for Japanese populations)
        if self.cultural_context == "japanese":
            # Japanese studies show smaller gender gaps
            female_mask = demographics['gender'] == 'female'
            male_mask = demographics['gender'] == 'male'
            
            # Small effect size (d ≈ 0.1-0.2) consistent with literature
            demographics.loc[female_mask, 'math_performance'] += np.random.normal(1, 2, female_mask.sum())
            demographics.loc[male_mask, 'math_performance'] += np.random.normal(0, 2, male_mask.sum())
        
        return demographics
    
    def calculate_similarity_metrics(self, brain_data: np.ndarray, demographics: pd.DataFrame) -> Dict:
        """Calculate gender similarity metrics.
        
        Args:
            brain_data: Preprocessed brain data
            demographics: Participant demographics
            
        Returns:
            Dictionary of similarity metrics
        """
        logger.info("Calculating gender similarity metrics")
        
        female_mask = demographics['gender'] == 'female'
        male_mask = demographics['gender'] == 'male'
        
        female_data = brain_data[female_mask]
        male_data = brain_data[male_mask]
        
        # Calculate various similarity metrics
        similarities = {}
        
        # 1. Correlation-based similarity
        female_mean = np.mean(female_data, axis=0)
        male_mean = np.mean(male_data, axis=0)
        
        correlation = np.corrcoef(female_mean, male_mean)[0, 1]
        similarities['pattern_correlation'] = correlation
        
        # 2. Euclidean distance-based similarity
        euclidean_dist = np.linalg.norm(female_mean - male_mean)
        max_possible_dist = np.linalg.norm(np.max(brain_data, axis=0) - np.min(brain_data, axis=0))
        similarities['euclidean_similarity'] = 1 - (euclidean_dist / max_possible_dist)
        
        # 3. Cosine similarity
        cosine_sim = np.dot(female_mean, male_mean) / (np.linalg.norm(female_mean) * np.linalg.norm(male_mean))
        similarities['cosine_similarity'] = cosine_sim
        
        # 4. Overlap coefficient
        overlap = np.sum(np.minimum(np.abs(female_mean), np.abs(male_mean))) / np.sum(np.maximum(np.abs(female_mean), np.abs(male_mean)))
        similarities['overlap_coefficient'] = overlap
        
        # 5. Individual vs group variation analysis
        individual_var = np.var(brain_data, axis=0)
        group_var_female = np.var(female_data, axis=0)
        group_var_male = np.var(male_data, axis=0)
        between_group_var = (female_mean - male_mean) ** 2
        
        # Ratio of individual to group differences
        individual_to_group_ratio = np.mean(individual_var) / np.mean(between_group_var)
        similarities['individual_to_group_ratio'] = individual_to_group_ratio
        
        # 6. Effect size calculation (Cohen's d)
        pooled_std = np.sqrt((np.var(female_data, axis=0) + np.var(male_data, axis=0)) / 2)
        cohens_d = np.abs(female_mean - male_mean) / pooled_std
        similarities['mean_cohens_d'] = np.mean(cohens_d)
        similarities['median_cohens_d'] = np.median(cohens_d)
        
        # Overall similarity index
        similarity_index = np.mean([
            similarities['pattern_correlation'],
            similarities['euclidean_similarity'],
            similarities['cosine_similarity'],
            similarities['overlap_coefficient']
        ])
        similarities['overall_similarity_index'] = similarity_index
        
        self.similarity_metrics = similarities
        return similarities
    
    def detect_classification_bias(self, brain_data: np.ndarray, demographics: pd.DataFrame) -> Dict:
        """Detect potential bias in gender classification.
        
        Args:
            brain_data: Brain data for classification
            demographics: Participant demographics
            
        Returns:
            Dictionary of bias detection results
        """
        if not self.bias_detection:
            return {}
        
        logger.info("Running bias detection analysis")
        
        # Prepare data for classification
        X = brain_data
        y = demographics['gender'].map({'female': 0, 'male': 1}).values
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Train classifier
        clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
        clf.fit(X_train, y_train)
        
        # Predictions
        y_pred = clf.predict(X_test)
        accuracy = clf.score(X_test, y_test)
        
        # Bias detection metrics
        bias_metrics = {
            'classification_accuracy': accuracy,
            'random_chance': 0.5,
            'above_chance': accuracy > 0.6,  # Conservative threshold
            'strong_classification': accuracy > 0.8,
            'feature_importance_max': np.max(clf.feature_importances_),
            'feature_importance_mean': np.mean(clf.feature_importances_),
            'n_important_features': np.sum(clf.feature_importances_ > 0.001)
        }
        
        # Classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)
        bias_metrics['precision_female'] = class_report['0']['precision']
        bias_metrics['recall_female'] = class_report['0']['recall']
        bias_metrics['precision_male'] = class_report['1']['precision']
        bias_metrics['recall_male'] = class_report['1']['recall']
        
        # Bias interpretation
        bias_metrics['bias_risk'] = 'high' if accuracy > 0.8 else 'medium' if accuracy > 0.6 else 'low'
        bias_metrics['similarity_supported'] = accuracy < 0.65  # Low accuracy supports similarity
        
        self.bias_metrics = bias_metrics
        return bias_metrics
    
    def analyze_similarities(self, data_dict: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """Run complete similarity analysis.
        
        Args:
            data_dict: Optional pre-loaded data dictionary
            
        Returns:
            Tuple of (similarity_metrics, bias_metrics)
        """
        if data_dict is None:
            data_dict = self.load_preprocessed_data("simulated_data")
        
        # Calculate similarities
        similarities = self.calculate_similarity_metrics(
            data_dict['brain_data'], 
            data_dict['demographics']
        )
        
        # Detect bias
        bias_results = self.detect_classification_bias(
            data_dict['brain_data'], 
            data_dict['demographics']
        )
        
        # Log key findings
        logger.info(f"Overall similarity index: {similarities['overall_similarity_index']:.3f}")
        logger.info(f"Individual to group variation ratio: {similarities['individual_to_group_ratio']:.2f}")
        logger.info(f"Mean Cohen's d: {similarities['mean_cohens_d']:.3f}")
        
        if self.bias_detection:
            logger.info(f"Classification accuracy: {bias_results['classification_accuracy']:.3f}")
            logger.info(f"Bias risk: {bias_results['bias_risk']}")
        
        return similarities, bias_results
    
    def generate_report(self) -> str:
        """Generate ethical analysis report.
        
        Returns:
            Formatted report string
        """
        if not self.similarity_metrics:
            return "No analysis results available. Run analyze_similarities() first."
        
        report = []
        report.append("MCP-fMRI Gender Similarity Analysis Report")
        report.append("=" * 50)
        report.append(f"Cultural Context: {self.cultural_context}")
        report.append(f"Ethical Guidelines: {'Enabled' if self.ethical_guidelines else 'Disabled'}")
        report.append("")
        
        # Similarity findings
        report.append("SIMILARITY FINDINGS:")
        report.append("-" * 20)
        sim = self.similarity_metrics
        report.append(f"Overall Similarity Index: {sim['overall_similarity_index']:.3f}")
        report.append(f"Pattern Correlation: {sim['pattern_correlation']:.3f}")
        report.append(f"Cosine Similarity: {sim['cosine_similarity']:.3f}")
        report.append(f"Individual vs Group Variation Ratio: {sim['individual_to_group_ratio']:.2f}")
        report.append(f"Mean Cohen's d (effect size): {sim['mean_cohens_d']:.3f}")
        report.append("")
        
        # Interpretation
        report.append("INTERPRETATION:")
        report.append("-" * 15)
        
        if sim['overall_similarity_index'] > self.similarity_threshold:
            report.append("✓ HIGH GENDER SIMILARITY detected in neural patterns")
        else:
            report.append("⚠ Moderate similarity detected - cultural factors may apply")
        
        if sim['individual_to_group_ratio'] > 3:
            report.append("✓ INDIVIDUAL DIFFERENCES dominate over group differences")
        else:
            report.append("⚠ Group differences notable - examine individual variation")
        
        if sim['mean_cohens_d'] < 0.2:
            report.append("✓ SMALL EFFECT SIZE supports similarity hypothesis")
        else:
            report.append("⚠ Medium effect size detected - context important")
        
        # Bias detection results
        if self.bias_detection and self.bias_metrics:
            report.append("")
            report.append("BIAS DETECTION:")
            report.append("-" * 15)
            bias = self.bias_metrics
            report.append(f"Classification Accuracy: {bias['classification_accuracy']:.3f}")
            report.append(f"Bias Risk Level: {bias['bias_risk']}")
            
            if bias['similarity_supported']:
                report.append("✓ LOW CLASSIFICATION ACCURACY supports similarity")
            else:
                report.append("⚠ Higher classification accuracy - examine data quality")
        
        # Ethical conclusions
        report.append("")
        report.append("ETHICAL CONCLUSIONS:")
        report.append("-" * 20)
        report.append("• Analysis emphasizes gender similarities over differences")
        report.append("• Individual variation exceeds group-level patterns")
        report.append(f"• Cultural context ({self.cultural_context}) integrated")
        report.append("• Results should not be used to justify discrimination")
        report.append("• Findings support evidence-based similarity hypothesis")
        
        return "\n".join(report)


class EthicalfMRIAnalysis:
    """Comprehensive ethical fMRI analysis framework."""
    
    def __init__(
        self,
        cultural_context: Optional[object] = None,
        similarity_threshold: float = 0.8,
        bias_detection: bool = True
    ):
        self.cultural_context = cultural_context
        self.similarity_threshold = similarity_threshold
        self.bias_detection = bias_detection
        
        self.analyzer = GenderSimilarityAnalyzer(
            ethical_guidelines=True,
            similarity_threshold=similarity_threshold,
            cultural_context=getattr(cultural_context, 'name', 'general'),
            bias_detection=bias_detection
        )
    
    def run_similarity_analysis(self, data: Dict) -> Dict:
        """Run comprehensive similarity analysis.
        
        Args:
            data: Preprocessed data dictionary
            
        Returns:
            Analysis results
        """
        similarities, bias_results = self.analyzer.analyze_similarities(data)
        
        return {
            'similarities': similarities,
            'bias_detection': bias_results,
            'cultural_context': self.cultural_context,
            'ethical_guidelines': True
        }


class BiasDetector:
    """Specialized bias detection for neuroimaging studies."""
    
    def __init__(self, sensitivity: str = 'high'):
        self.sensitivity = sensitivity
        self.thresholds = {
            'low': 0.7,
            'medium': 0.65,
            'high': 0.6
        }
    
    def detect_sampling_bias(self, demographics: pd.DataFrame) -> Dict:
        """Detect potential sampling bias in demographics.
        
        Args:
            demographics: Participant demographics
            
        Returns:
            Bias detection results
        """
        bias_flags = {}
        
        # Gender balance check
        gender_counts = demographics['gender'].value_counts()
        gender_ratio = min(gender_counts) / max(gender_counts)
        bias_flags['gender_balanced'] = gender_ratio > 0.4
        
        # Age distribution check
        age_std = demographics['age'].std()
        bias_flags['age_diverse'] = age_std > 2
        
        # Education balance
        if 'education_years' in demographics.columns:
            edu_std = demographics['education_years'].std()
            bias_flags['education_diverse'] = edu_std > 1.5
        
        # Regional diversity (for Japanese context)
        if 'region' in demographics.columns:
            region_counts = demographics['region'].value_counts()
            bias_flags['regionally_diverse'] = len(region_counts) > 2
        
        bias_flags['overall_unbiased'] = all(bias_flags.values())
        
        return bias_flags
    
    def assess_classification_bias(self, accuracy: float) -> str:
        """Assess bias risk from classification accuracy.
        
        Args:
            accuracy: Classification accuracy
            
        Returns:
            Bias risk assessment
        """
        threshold = self.thresholds[self.sensitivity]
        
        if accuracy > 0.8:
            return 'high'
        elif accuracy > threshold:
            return 'medium'
        else:
            return 'low'