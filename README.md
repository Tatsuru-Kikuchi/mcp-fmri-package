# MCP-fMRI: Ethical Analysis of Mathematical Abilities Using Generative AI

> **Performance analysis in mathematical skills with emphasis on gender similarities using fMRI neuroimaging data and generative AI techniques - focused on Japanese populations**
> 
> **🆕 NEW: Comprehensive Economic Bias Detection for Economics Research**

[![PyPI version](https://badge.fury.io/py/mcp-fmri.svg)](https://badge.fury.io/py/mcp-fmri)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 🎯 Research Objectives

This package provides tools for investigating the neural mechanisms underlying mathematical cognition using advanced generative AI techniques applied to functional MRI (fMRI) data. **Our approach emphasizes similarities over differences** and promotes ethical AI practices in neuroimaging research.

### 🆕 **NEW: Economic Bias Detection**

**Addressing a Critical Gap in Economic Research**: Economic studies using neuroimaging data are particularly vulnerable to bias that can lead to incorrect policy decisions and perpetuate inequality. MCP-fMRI now includes comprehensive economic bias detection specifically designed for economics researchers.

### Primary Goals
- **Investigate neural mechanisms** underlying mathematical cognition in Japanese populations
- **Emphasize gender similarities** rather than differences in mathematical abilities
- **🆕 Detect and mitigate economic bias** in neuroimaging research applications
- **Apply generative AI** to enhance fMRI analysis capabilities and address small dataset challenges
- **Integrate cultural context** specific to Japanese sociocultural factors
- **Establish ethical guidelines** for AI-based research in neuroimaging and economics

### Research Philosophy
We follow evidence-based approaches showing that gender similarities dominate mathematical cognition at the neural level, with special attention to economic factors that can bias research and policy.

## 🚨 Economic Bias: A Critical Problem

### Why Economic Bias Matters

Bias in economic research can:
- **Lead to incorrect policy decisions** affecting millions
- **Perpetuate economic inequalities** through biased research
- **Undermine scientific credibility** in policy-relevant research
- **Justify discrimination** in hiring, education, and investment

### Types of Economic Bias MCP-fMRI Detects

1. **🎯 Selection Bias**: Non-representative samples by income/education
2. **💰 SES Confounding**: Socioeconomic status affects both brain and behavior
3. **📅 Temporal Bias**: Economic conditions during data collection
4. **🤖 Algorithmic Bias**: Models discriminating by economic factors
5. **🏛️ Policy Misuse**: Research used to justify economic discrimination

## 🚀 Installation

### From PyPI (Recommended)

```bash
pip install mcp-fmri
```

### Optional Dependencies

```bash
# For neuroimaging support
pip install mcp-fmri[neuroimaging]

# For dashboard functionality
pip install mcp-fmri[dash]

# For development
pip install mcp-fmri[dev]
```

## 🔧 Quick Economic Bias Check

**Before conducting any economic analysis**, run a quick bias check:

```python
from mcp_fmri import quick_economic_bias_check
import pandas as pd

# Load your demographics data
demographics = pd.read_csv('your_demographics.csv')

# Quick bias check
bias_check = quick_economic_bias_check(demographics)

print(bias_check['summary'])
for warning in bias_check['warnings']:
    print(warning)
for rec in bias_check['recommendations']:
    print(f"📋 {rec}")
```

## 💡 Economic Bias Detection Examples

### 1. Comprehensive Economic Bias Analysis

```python
from mcp_fmri.economic_bias import (
    EconomicBiasDetector,
    run_comprehensive_economic_bias_analysis
)

# Initialize detector for economics research
detector = EconomicBiasDetector(
    sensitivity='high',  # Strict bias detection
    economic_context='japanese'  # Cultural context
)

# Detect selection bias
selection_bias = detector.detect_selection_bias_economic(demographics)

if selection_bias['economic_selection_bias']:
    print("⚠️ Economic selection bias detected!")
    print(f"Bias severity: {selection_bias['bias_severity']:.2f}")
    print("Recommendation: Implement stratified sampling")

# Detect SES confounding
ses_confounding = detector.detect_ses_confounding(
    demographics, brain_data, performance_data
)

if ses_confounding['ses_brain_confounding']:
    print("⚠️ SES confounding detected!")
    print(f"SES-brain correlation: {ses_confounding['ses_brain_correlation']:.3f}")
    print("Recommendation: Control for SES in all analyses")
```

### 2. Economic Policy-Safe Analysis

```python
from mcp_fmri import (
    GenderSimilarityAnalyzer,
    JapaneseCulturalContext
)
from mcp_fmri.economic_bias import EconomicBiasDetector

# Initialize with economic bias protection
analyzer = GenderSimilarityAnalyzer(
    ethical_guidelines=True,
    bias_detection=True,  # Includes economic bias detection
    cultural_context="japanese"
)

# Japanese cultural & economic context
context = JapaneseCulturalContext()
economic_detector = EconomicBiasDetector(economic_context='japanese')

# Load and validate data
data_dict = analyzer.load_preprocessed_data("your_data_path")

# Check for economic bias BEFORE analysis
economic_bias = run_comprehensive_economic_bias_analysis(
    data_dict, economic_context='japanese'
)

if economic_bias['overall_assessment']['bias_detected']:
    print(f"🚨 Economic bias detected: {economic_bias['overall_assessment']['bias_level']}")
    print("Applying bias mitigation before proceeding...")
    
    # Apply recommended mitigations
    for rec in economic_bias['comprehensive_recommendations']:
        print(f"📋 {rec}")
    
    # DO NOT proceed with policy recommendations until bias is addressed
    print("⚠️ Address bias before making policy recommendations")
else:
    # Safe to proceed with analysis
    similarities, bias_results = analyzer.analyze_similarities(data_dict)
    print(f"✅ Analysis complete - safe for policy considerations")
```

### 3. Economics-Specific Applications

```python
# For economics researchers
from mcp_fmri.economic_bias import EconomicBiasDetector

# Assess workforce potential without discrimination
def assess_workforce_potential(population_data):
    detector = EconomicBiasDetector(sensitivity='high')
    
    # Check for bias in workforce assessment
    bias_flags = detector.detect_selection_bias_economic(
        population_data['demographics']
    )
    
    if bias_flags['economic_selection_bias']:
        return {
            'status': 'bias_detected',
            'recommendation': 'Expand sampling before workforce assessment',
            'bias_severity': bias_flags['bias_severity']
        }
    
    # Safe to proceed with individual-focused assessment
    individual_assessments = []
    for participant in population_data['participants']:
        # Focus on individual potential, not group predictions
        potential = assess_individual_potential(participant)
        individual_assessments.append(potential)
    
    return {
        'status': 'assessment_complete',
        'individual_assessments': individual_assessments,
        'avoid_group_generalizations': True
    }

# Educational resource allocation
def allocate_education_resources(school_data, budget):
    detector = EconomicBiasDetector(economic_context='japanese')
    
    allocation_results = {}
    for school_id, data in school_data.items():
        # Check for bias in needs assessment
        bias_check = detector.detect_selection_bias_economic(data['demographics'])
        
        if bias_check['economic_selection_bias']:
            print(f"⚠️ Bias detected in {school_id} needs assessment")
            # Apply bias correction before allocation
            needs = apply_bias_corrected_assessment(data)
        else:
            needs = assess_educational_needs(data)
        
        allocation_results[school_id] = calculate_fair_allocation(needs, budget)
    
    return allocation_results
```

## 📊 Key Features

### 🧠 **Neural Analysis** (Enhanced)
- Similarity-focused fMRI preprocessing pipeline
- Gender similarity detection models
- **🆕 Economic bias-aware analysis**
- Cultural context integration
- Comprehensive bias detection and mitigation

### 💰 **Economic Bias Detection** (NEW)
- **Selection bias detection** by income, education, employment
- **SES confounding analysis** with brain and performance data
- **Temporal economic bias** detection during study periods
- **Algorithmic bias** detection in predictive models
- **Policy misuse prevention** safeguards

### 🌏 **Cultural Considerations**
- Japanese educational system context
- **🆕 Japanese economic context** (collectivist values, employment patterns)
- Collectivist culture framework
- Stereotype acquisition patterns
- Regional economic diversity analysis

### 🤖 **Generative AI Components**
- Variational Autoencoders (VAE) for feature learning
- **🆕 Bias-aware data augmentation**
- Transfer learning across populations
- Economic confound-controlled synthetic data generation

### 📊 **Ethical Framework** (Enhanced)
- Similarity emphasis over differences
- **🆕 Economic discrimination prevention**
- Individual focus over group generalizations
- Comprehensive bias mitigation techniques
- **🆕 Policy-safe reporting** guidelines
- Cultural sensitivity integration

## 📈 Economic Research Applications

### 1. **Educational Policy** 💎
- Bias-free assessment of educational interventions
- Fair resource allocation across schools
- Individual-focused rather than group-based policies

### 2. **Workforce Development** 🏢
- Ethical assessment of mathematical potential
- Discrimination-free hiring tool development
- Individual career pathway optimization

### 3. **Economic Development** 🌟
- Regional capacity assessment with bias controls
- Evidence-based investment recommendations
- Cultural context-aware development planning

## 🛡️ Bias Prevention Safeguards

### Automated Safeguards
```python
# Automatic bias checking
from mcp_fmri import set_config

# Enable strict bias monitoring
set_config(
    ethical_guidelines=True,
    economic_bias_detection=True,
    auto_bias_check=True  # Automatically check for bias
)

# All analyses now include automatic bias detection
analyzer = GenderSimilarityAnalyzer()  # Automatically bias-aware
```

### Policy Application Safeguards
```python
from mcp_fmri.utils import validate_policy_application

# Before using research for policy
policy_validation = validate_policy_application(
    research_results=your_results,
    intended_use='educational_resource_allocation',
    population='japanese_students'
)

if not policy_validation['safe_for_policy']:
    print("🚨 UNSAFE for policy application")
    for issue in policy_validation['issues']:
        print(f"⚠️ {issue}")
else:
    print("✅ Safe for policy application with safeguards")
```

## 📖 Command Line Interface

```bash
# Preprocess with economic bias checking
mcp-fmri-preprocess --input /data/raw --output /data/processed \
                    --economic-bias-check --cultural-context japanese

# Analyze with comprehensive bias detection
mcp-fmri-analyze --data /data/processed --output /results \
                 --economic-bias-detection --policy-safe-reporting
```

## 🧪 Testing Your Data for Economic Bias

```python
# Test your existing research for economic bias
from mcp_fmri.economic_bias import run_comprehensive_economic_bias_analysis

# Load your data
data_dict = {
    'demographics': your_demographics_df,
    'brain_data': your_brain_data,
    'performance_data': your_performance_scores,
    'study_timeline': your_study_dates
}

# Comprehensive bias analysis
bias_results = run_comprehensive_economic_bias_analysis(
    data_dict, 
    economic_context='japanese'
)

print(f"Overall bias level: {bias_results['overall_assessment']['bias_level']}")
print(f"Bias score: {bias_results['overall_assessment']['bias_score']:.3f}")

if bias_results['overall_assessment']['bias_detected']:
    print("\n🚨 BIAS DETECTED - Recommendations:")
    for rec in bias_results['comprehensive_recommendations']:
        print(f"📋 {rec}")
else:
    print("\n✅ No significant economic bias detected")
```

## 📚 Documentation

- **[Economic Bias Guide](docs/bias_reduction_economics.md)** - Comprehensive bias detection guide
- **[Economic Implications](docs/economic_implications.md)** - Policy and economic applications
- **[Quick Start Guide](docs/quickstart.md)** - Get started in 5 minutes
- **[Ethical Guidelines](docs/ethical_guidelines.md)** - Complete ethical framework
- **[API Documentation](docs/api.md)** - Detailed API reference

## 🔬 Scientific Background

### Economic Bias Research Context

Recent studies highlight the critical importance of bias detection in economic research:

- **Economic inequality effects**: SES significantly affects both neural development and mathematical performance, creating spurious correlations
- **Policy implications**: Biased neuroimaging research has been used to justify discriminatory educational and hiring practices
- **Cultural factors**: Economic systems and cultural values interact, requiring context-specific analysis
- **Methodological gaps**: Most neuroimaging studies lack comprehensive economic bias detection

### Original Research Foundation

- **Gender similarities hypothesis**: Large-scale studies show more similarities than differences in mathematical cognition ([Hyde et al., 2008](https://science.sciencemag.org/content/321/5888/494))
- **Neural similarity findings**: fMRI studies in children (3-10 years) show no significant gender differences in mathematical brain activation ([Kersey et al., 2019](https://www.nature.com/articles/s41539-019-0057-x))
- **Cultural factors**: Japanese children may acquire gender stereotypes later than Western populations, suggesting environmental influences ([Tatsuno et al., 2022](https://www.nature.com/articles/s41598-022-20815-2))

## ⚠️ Critical Warnings for Economics Researchers

### 🚨 **NEVER Use For Discrimination**
```python
# ❌ WRONG - Group-based decisions
if participant['gender'] == 'female':
    predicted_math_ability = model.predict_low()
    
# ✅ CORRECT - Individual assessment
potential = assess_individual_potential(
    participant['brain_data'],
    control_for_bias=True,
    cultural_context='japanese'
)
```

### 🛡️ **Always Check Bias First**
```python
# ❌ WRONG - Direct analysis without bias check
results = analyze_without_bias_check(data)
make_policy_recommendations(results)  # DANGEROUS!

# ✅ CORRECT - Bias-aware workflow
bias_check = quick_economic_bias_check(data)
if bias_check['warnings']:
    print("Address bias before proceeding")
    apply_bias_mitigation(data)

results = analyze_with_bias_controls(data)
make_responsible_recommendations(results)
```

### 📊 **Focus on Individual Differences**
```python
# ❌ WRONG - Group-level generalizations
print(f"Group X performs {mean_diff:.2f} points lower")

# ✅ CORRECT - Individual variation emphasis
print(f"Individual differences ({individual_std:.2f}) exceed")
print(f"group differences ({group_diff:.2f}) by {ratio:.1f}:1")
print("Focus on individual assessment and support")
```

## 🤝 Contributing

We welcome contributions that align with our ethical research principles, especially:

- **🔍 Economic bias detection improvements**
- **🌍 Additional cultural contexts**
- **📊 Policy-safe analysis methods**
- **🛡️ Bias mitigation techniques**
- **📖 Documentation and examples**

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Economics research community** for highlighting bias concerns
- **Neuroimaging community** for open science practices
- **Ethical AI researchers** for bias mitigation frameworks
- **Japanese research institutions** for cultural context insights
- **Policy researchers** emphasizing responsible research translation

## 📞 Support

- **Documentation**: [https://tatsuru-kikuchi.github.io/MCP-fMRI/](https://tatsuru-kikuchi.github.io/MCP-fMRI/)
- **Issues**: [GitHub Issues](https://github.com/Tatsuru-Kikuchi/mcp-fmri-package/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Tatsuru-Kikuchi/mcp-fmri-package/discussions)
- **Economic Research Questions**: Tag issues with `economics` or `bias-detection`

---

## 🎯 **Quick Start for Economics Researchers**

```python
# Install
pip install mcp-fmri

# Quick bias check
from mcp_fmri import quick_economic_bias_check
bias_result = quick_economic_bias_check(your_demographics)
print(bias_result['summary'])

# If bias detected, see docs/bias_reduction_economics.md
# If no bias, proceed with analysis
from mcp_fmri import GenderSimilarityAnalyzer
analyzer = GenderSimilarityAnalyzer(ethical_guidelines=True)
results = analyzer.analyze_similarities(your_data)

# Generate policy-safe report
report = analyzer.generate_report()
print(report)
```

---

**⚠️ Important Note**: This research emphasizes gender similarities in mathematical cognition and includes comprehensive economic bias detection. Results should never be used to justify discrimination or reinforce stereotypes. All findings should be interpreted within their cultural and economic context with emphasis on individual differences over group generalizations.

**🏛️ Policy Note**: Before using any findings for policy decisions, ensure comprehensive bias analysis has been conducted and appropriate safeguards are in place to prevent discriminatory applications.
