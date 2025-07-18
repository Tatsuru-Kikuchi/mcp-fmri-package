#!/usr/bin/env python3
"""
MCP-fMRI Advanced Analysis Example
Demonstrates advanced features including bias detection and cultural interpretation
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json

from mcp_fmri import (
    EthicalfMRIAnalysis,
    BiasDetector,
    JapaneseCulturalContext,
    load_config,
    validate_data,
    ethical_guidelines,
    check_ethical_compliance
)

def create_advanced_config():
    """Create advanced analysis configuration."""
    config = {
        'analysis': {
            'similarity_threshold': 0.85,
            'bias_detection_sensitivity': 'high',
            'cultural_context': 'japanese',
            'individual_emphasis': True,
            'similarity_focus': True
        },
        'cultural_settings': {
            'education_system': 'collectivist',
            'stereotype_timing': 'late',
            'regional_diversity': True
        },
        'visualization': {
            'interactive_dashboard': True,
            'publication_quality': True,
            'color_palette': 'viridis'
        },
        'ethical_guidelines': {
            'bias_detection': True,
            'individual_emphasis': True,
            'cultural_sensitivity': True,
            'transparency': True
        },
        'output': {
            'save_plots': True,
            'generate_report': True,
            'export_data': True
        }
    }
    return config

def demonstrate_bias_detection():
    """Demonstrate comprehensive bias detection."""
    print("\n🔍 ADVANCED BIAS DETECTION DEMONSTRATION")
    print("=" * 45)
    
    # Initialize bias detector with high sensitivity
    detector = BiasDetector(sensitivity='high')
    
    # Create sample demographics with known biases
    print("Creating sample with potential biases...")
    
    # Unbalanced sample
    unbalanced_demographics = pd.DataFrame({
        'gender': ['female'] * 80 + ['male'] * 20,  # 80/20 split
        'age': np.random.normal(21, 2, 100),
        'education_years': [16] * 100,  # No variation
        'region': ['tokyo'] * 100  # No regional diversity
    })
    
    bias_flags = detector.detect_sampling_bias(unbalanced_demographics)
    
    print("\nBias Detection Results:")
    print(f"  Gender balanced: {bias_flags['gender_balanced']}")
    print(f"  Age diverse: {bias_flags['age_diverse']}")
    print(f"  Education diverse: {bias_flags['education_diverse']}")
    print(f"  Regionally diverse: {bias_flags['regionally_diverse']}")
    print(f"  Overall unbiased: {bias_flags['overall_unbiased']}")
    
    if not bias_flags['overall_unbiased']:
        print("\n⚠️  BIAS DETECTED: Sample shows multiple bias indicators")
        print("Recommendations:")
        if not bias_flags['gender_balanced']:
            print("  - Improve gender balance in recruitment")
        if not bias_flags['regionally_diverse']:
            print("  - Include participants from multiple regions")
        if not bias_flags['education_diverse']:
            print("  - Include varied educational backgrounds")
    
    # Now create a balanced sample
    print("\nCreating balanced sample...")
    balanced_demographics = pd.DataFrame({
        'gender': ['female'] * 50 + ['male'] * 50,
        'age': np.random.normal(21, 3, 100),
        'education_years': np.random.normal(15, 2, 100),
        'region': np.random.choice(['tokyo', 'osaka', 'kyoto', 'other'], 100)
    })
    
    balanced_flags = detector.detect_sampling_bias(balanced_demographics)
    print(f"\nBalanced sample - Overall unbiased: {balanced_flags['overall_unbiased']}")
    
    return balanced_demographics

def demonstrate_cultural_analysis(demographics):
    """Demonstrate cultural context analysis."""
    print("\n🎌 CULTURAL CONTEXT ANALYSIS")
    print("=" * 30)
    
    # Initialize Japanese cultural context
    context = JapaneseCulturalContext(
        education_system="collectivist",
        stereotype_timing="late",
        regional_diversity=True
    )
    
    # Validate sample against Japanese expectations
    print("Validating sample for Japanese population...")
    validation = context.validate_sample(demographics)
    
    print("\nCultural Validation Results:")
    for key, value in validation.items():
        if key != 'sample_representative':
            print(f"  {key.replace('_', ' ').title()}: {value}")
    
    print(f"\nOverall Assessment: {'✅ Representative' if validation['sample_representative'] else '❌ Not Representative'}")
    
    # Get cultural adjustments
    adjustments = context.get_cultural_adjustments()
    print("\nCultural Adjustments for Japanese Context:")
    print(f"  Expected effect size: {adjustments['expected_effect_size']} (small)")
    print(f"  Expected similarity: {adjustments['similarity_expectation']} (high)")
    print(f"  Stereotype onset age: {adjustments['stereotype_age_onset']} years (late)")
    print(f"  Educational equality: {adjustments['educational_equality']} (high)")
    
    return context, adjustments

def demonstrate_ethical_compliance():
    """Demonstrate ethical compliance checking."""
    print("\n⚖️ ETHICAL COMPLIANCE DEMONSTRATION")
    print("=" * 35)
    
    # Show ethical guidelines
    guidelines = ethical_guidelines()
    print("Core Ethical Guidelines:")
    for i, (key, description) in enumerate(guidelines.items(), 1):
        print(f"  {i}. {key.replace('_', ' ').title()}:")
        print(f"     {description}")
        if i >= 3:  # Show first 3 for brevity
            print(f"     ... and {len(guidelines) - 3} more guidelines")
            break
    
    # Check compliance of different configurations
    print("\nChecking Ethical Compliance:")
    
    # Good configuration
    good_config = {
        'similarity_focus': True,
        'cultural_context': 'japanese',
        'bias_detection': True,
        'individual_emphasis': True,
        'documentation_level': 'detailed',
        'diverse_sampling': True
    }
    
    good_compliance = check_ethical_compliance(good_config)
    print(f"\n✅ Well-designed study: {good_compliance['percentage']:.1f}% compliant")
    
    # Poor configuration
    poor_config = {
        'similarity_focus': False,
        'cultural_context': None,
        'bias_detection': False,
        'individual_emphasis': False,
        'documentation_level': 'minimal',
        'diverse_sampling': False
    }
    
    poor_compliance = check_ethical_compliance(poor_config)
    print(f"❌ Poorly-designed study: {poor_compliance['percentage']:.1f}% compliant")
    
    if poor_compliance['recommendations']:
        print("\nRecommendations for improvement:")
        for rec in poor_compliance['recommendations'][:3]:
            print(f"  • {rec}")

def run_comprehensive_analysis(demographics, cultural_context):
    """Run comprehensive analysis with all advanced features."""
    print("\n🧠 COMPREHENSIVE SIMILARITY ANALYSIS")
    print("=" * 38)
    
    # Initialize comprehensive analysis framework
    analysis = EthicalfMRIAnalysis(
        cultural_context=cultural_context,
        similarity_threshold=0.85,
        bias_detection=True
    )
    
    # Simulate brain data
    print("Generating simulated brain data...")
    n_participants = len(demographics)
    n_voxels = 10000
    
    # Create realistic brain data with subtle patterns
    brain_data = np.random.normal(0, 1, (n_participants, n_voxels))
    
    # Add spatial correlations
    for i in range(1, min(1000, n_voxels)):
        brain_data[:, i] = 0.3 * brain_data[:, i-1] + 0.7 * brain_data[:, i]
    
    # Add very subtle gender-related patterns (realistic small effect)
    female_mask = demographics['gender'] == 'female'
    male_mask = demographics['gender'] == 'male'
    
    # Small effect in a subset of voxels
    effect_voxels = np.random.choice(n_voxels, 100, replace=False)
    effect_size = 0.15  # Small effect consistent with literature
    
    brain_data[female_mask][:, effect_voxels] += np.random.normal(0, effect_size, (female_mask.sum(), len(effect_voxels)))
    brain_data[male_mask][:, effect_voxels] += np.random.normal(0, effect_size, (male_mask.sum(), len(effect_voxels)))
    
    data_dict = {
        'brain_data': brain_data,
        'demographics': demographics,
        'n_participants': n_participants,
        'n_voxels': n_voxels
    }
    
    # Validate data
    print("Validating data quality...")
    brain_validation = validate_data(brain_data, 'neuroimaging')
    demo_validation = validate_data(demographics, 'demographics')
    
    print(f"Brain data valid: {brain_validation['valid']}")
    print(f"Demographics valid: {demo_validation['valid']}")
    
    if brain_validation['warnings']:
        print(f"Brain data warnings: {brain_validation['warnings']}")
    if demo_validation['warnings']:
        print(f"Demographics warnings: {demo_validation['warnings']}")
    
    # Run analysis
    print("\nRunning similarity analysis...")
    results = analysis.run_similarity_analysis(data_dict)
    
    # Extract results
    similarities = results['similarities']
    bias_detection = results['bias_detection']
    
    # Cultural interpretation
    cultural_interpretation = cultural_context.interpret_results(
        similarities, bias_detection
    )
    
    print("\n📊 ANALYSIS RESULTS:")
    print("-" * 20)
    print(f"Overall Similarity Index: {similarities['overall_similarity_index']:.3f}")
    print(f"Individual:Group Ratio: {similarities['individual_to_group_ratio']:.2f}:1")
    print(f"Mean Effect Size: {similarities['mean_cohens_d']:.3f}")
    print(f"Classification Accuracy: {bias_detection['classification_accuracy']:.3f}")
    print(f"Bias Risk: {bias_detection['bias_risk']}")
    
    print("\n🎌 CULTURAL INTERPRETATION:")
    print("-" * 28)
    print(f"Similarity Assessment: {cultural_interpretation['similarity_assessment']}")
    print(f"Effect Size Assessment: {cultural_interpretation['effect_size_assessment']}")
    print(f"Individual Variation: {cultural_interpretation['individual_variation_assessment']}")
    print(f"Bias Assessment: {cultural_interpretation['bias_assessment']}")
    
    print("\n📝 CULTURAL CONCLUSION:")
    print("-" * 23)
    print(cultural_interpretation['cultural_conclusion'])
    
    return results, cultural_interpretation

def save_advanced_results(results, cultural_interpretation, output_dir):
    """Save comprehensive results."""
    print("\n💾 SAVING COMPREHENSIVE RESULTS")
    print("=" * 32)
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save detailed results
    detailed_results = {
        'analysis_results': results,
        'cultural_interpretation': cultural_interpretation,
        'analysis_metadata': {
            'version': '0.1.0',
            'analysis_type': 'comprehensive_similarity',
            'cultural_context': 'japanese',
            'ethical_guidelines': True
        }
    }
    
    # Convert to JSON-serializable format
    json_results = {}
    for key, value in detailed_results.items():
        if key == 'analysis_results':
            json_results[key] = {
                'similarities': {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                               for k, v in value['similarities'].items()},
                'bias_detection': {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                                 for k, v in value['bias_detection'].items()}
            }
        else:
            json_results[key] = value
    
    with open(f"{output_dir}/comprehensive_results.json", 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"Results saved to {output_dir}/comprehensive_results.json")
    
    # Create summary report
    summary = f"""
MCP-fMRI Advanced Analysis Summary
=================================

Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Cultural Context: Japanese
Ethical Guidelines: Enabled

KEY FINDINGS:
• Overall Similarity: {results['similarities']['overall_similarity_index']:.3f}
• Individual:Group Ratio: {results['similarities']['individual_to_group_ratio']:.2f}:1
• Effect Size: {results['similarities']['mean_cohens_d']:.3f}
• Classification Accuracy: {results['bias_detection']['classification_accuracy']:.3f}
• Bias Risk: {results['bias_detection']['bias_risk']}

CULTURAL ASSESSMENT:
• Similarity: {cultural_interpretation['similarity_assessment']}
• Effect Size: {cultural_interpretation['effect_size_assessment']}
• Individual Variation: {cultural_interpretation['individual_variation_assessment']}

CONCLUSION:
{cultural_interpretation['cultural_conclusion']}

ETHICAL REMINDER:
These findings emphasize gender similarities and should never be used
to justify discrimination or reinforce stereotypes.
"""
    
    with open(f"{output_dir}/analysis_summary.txt", 'w', encoding='utf-8') as f:
        f.write(summary)
    
    print(f"Summary saved to {output_dir}/analysis_summary.txt")

def main():
    """Main advanced analysis demonstration."""
    print("🚀 MCP-fMRI ADVANCED ANALYSIS DEMONSTRATION")
    print("=" * 45)
    print("This example demonstrates advanced features including:")
    print("• Comprehensive bias detection")
    print("• Cultural context integration")
    print("• Ethical compliance checking")
    print("• Advanced similarity analysis")
    print()
    
    # 1. Demonstrate bias detection
    demographics = demonstrate_bias_detection()
    
    # 2. Demonstrate cultural analysis
    cultural_context, adjustments = demonstrate_cultural_analysis(demographics)
    
    # 3. Demonstrate ethical compliance
    demonstrate_ethical_compliance()
    
    # 4. Run comprehensive analysis
    results, cultural_interpretation = run_comprehensive_analysis(demographics, cultural_context)
    
    # 5. Save results
    output_dir = "./advanced_results"
    save_advanced_results(results, cultural_interpretation, output_dir)
    
    print("\n🎉 ADVANCED ANALYSIS COMPLETE!")
    print("=" * 32)
    print("This demonstration showcased:")
    print("✅ Multi-level bias detection")
    print("✅ Cultural context integration")
    print("✅ Ethical compliance monitoring")
    print("✅ Comprehensive similarity analysis")
    print("✅ Japanese population-specific interpretation")
    print()
    print("Key takeaway: MCP-fMRI provides a complete framework")
    print("for ethical neuroimaging research that emphasizes")
    print("similarities while respecting cultural context.")
    print()
    print(f"All results saved to: {output_dir}")

if __name__ == '__main__':
    main()