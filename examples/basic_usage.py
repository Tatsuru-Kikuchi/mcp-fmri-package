#!/usr/bin/env python3
"""
MCP-fMRI Basic Usage Example
Demonstrates basic preprocessing and analysis workflow
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Import MCP-fMRI modules
from mcp_fmri import (
    fMRIPreprocessor,
    GenderSimilarityAnalyzer,
    JapaneseCulturalContext,
    SimilarityPlotter,
    EthicalReportGenerator
)

def main():
    """Main example workflow."""
    print("MCP-fMRI Basic Usage Example")
    print("=" * 40)
    
    # 1. Setup directories (in practice, these would be your actual data paths)
    raw_data_dir = "./example_data/raw"
    processed_data_dir = "./example_data/processed"
    results_dir = "./example_results"
    
    # Create directories
    for dir_path in [raw_data_dir, processed_data_dir, results_dir]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print(f"Created example directories:")
    print(f"  Raw data: {raw_data_dir}")
    print(f"  Processed: {processed_data_dir}")
    print(f"  Results: {results_dir}")
    print()
    
    # 2. Initialize cultural context
    print("Initializing Japanese cultural context...")
    cultural_context = JapaneseCulturalContext(
        education_system="collectivist",
        stereotype_timing="late",
        regional_diversity=True
    )
    
    # Get cultural adjustments
    adjustments = cultural_context.get_cultural_adjustments()
    print(f"Expected effect size: {adjustments['expected_effect_size']}")
    print(f"Expected similarity: {adjustments['similarity_expectation']}")
    print()
    
    # 3. Preprocessing (simulated)
    print("Running fMRI preprocessing...")
    preprocessor = fMRIPreprocessor(
        raw_data_dir=raw_data_dir,
        output_dir=processed_data_dir,
        ethical_guidelines=True,
        cultural_context="japanese"
    )
    
    # In a real scenario, you would have actual participant data
    # For this example, we'll simulate the preprocessing step
    participant_list = [f"JP{i:03d}" for i in range(1, 21)]  # 20 participants
    
    print(f"Simulating preprocessing for {len(participant_list)} participants...")
    # Note: This would fail with real data since we don't have actual files
    # results = preprocessor.preprocess_batch(participant_list)
    print("Preprocessing simulation complete.")
    print()
    
    # 4. Gender similarity analysis
    print("Running gender similarity analysis...")
    analyzer = GenderSimilarityAnalyzer(
        ethical_guidelines=True,
        similarity_threshold=0.8,
        cultural_context="japanese",
        bias_detection=True
    )
    
    # Load or simulate preprocessed data
    print("Loading preprocessed data...")
    data_dict = analyzer.load_preprocessed_data(processed_data_dir)
    
    print(f"Data shape: {data_dict['brain_data'].shape}")
    print(f"Participants: {data_dict['n_participants']}")
    print(f"Voxels: {data_dict['n_voxels']}")
    print()
    
    # Validate sample with cultural context
    print("Validating sample representativeness...")
    validation = cultural_context.validate_sample(data_dict['demographics'])
    
    print(f"Sample representative: {validation['sample_representative']}")
    print(f"Gender balanced: {validation['gender_balanced']}")
    print(f"Regionally diverse: {validation['regionally_diverse']}")
    print()
    
    # Run similarity analysis
    print("Calculating similarity metrics...")
    similarities, bias_results = analyzer.analyze_similarities(data_dict)
    
    # Display key results
    print("KEY RESULTS:")
    print("-" * 12)
    print(f"Overall similarity index: {similarities['overall_similarity_index']:.3f}")
    print(f"Individual:Group ratio: {similarities['individual_to_group_ratio']:.2f}:1")
    print(f"Mean effect size (Cohen's d): {similarities['mean_cohens_d']:.3f}")
    print(f"Classification accuracy: {bias_results['classification_accuracy']:.3f}")
    print(f"Bias risk: {bias_results['bias_risk']}")
    print()
    
    # 5. Cultural interpretation
    print("Applying cultural interpretation...")
    cultural_interpretation = cultural_context.interpret_results(
        similarities, bias_results
    )
    
    print(f"Similarity assessment: {cultural_interpretation['similarity_assessment']}")
    print(f"Effect size assessment: {cultural_interpretation['effect_size_assessment']}")
    print(f"Individual variation: {cultural_interpretation['individual_variation_assessment']}")
    print()
    
    # 6. Generate visualizations
    print("Generating visualizations...")
    plotter = SimilarityPlotter()
    
    # Create similarity matrix plot
    fig1 = plotter.plot_similarity_matrix(similarities)
    fig1.savefig(f"{results_dir}/similarity_matrix.png", dpi=300, bbox_inches='tight')
    
    # Create individual vs group variation plot
    fig2 = plotter.plot_individual_vs_group_variation(similarities)
    fig2.savefig(f"{results_dir}/individual_vs_group.png", dpi=300, bbox_inches='tight')
    
    # Create effect size plot
    fig3 = plotter.plot_effect_size_distribution(similarities)
    fig3.savefig(f"{results_dir}/effect_size.png", dpi=300, bbox_inches='tight')
    
    # Create interactive dashboard
    fig4 = plotter.create_interactive_similarity_dashboard(similarities, bias_results)
    fig4.write_html(f"{results_dir}/interactive_dashboard.html")
    
    print(f"Plots saved to {results_dir}/")
    print()
    
    # 7. Generate comprehensive report
    print("Generating comprehensive report...")
    report_generator = EthicalReportGenerator(cultural_context="japanese")
    
    analysis_results = {
        'similarities': similarities,
        'bias_detection': bias_results,
        'cultural_interpretation': cultural_interpretation,
        'cultural_context': cultural_context
    }
    
    report = report_generator.generate_similarity_report(
        analysis_results,
        output_dir=results_dir
    )
    
    # Save text report
    with open(f"{results_dir}/analysis_report.txt", 'w', encoding='utf-8') as f:
        f.write(report['text_report'])
    
    print(f"Report saved to {results_dir}/analysis_report.txt")
    print()
    
    # 8. Display ethical conclusions
    print("ETHICAL CONCLUSIONS:")
    print("-" * 19)
    print(cultural_interpretation['cultural_conclusion'])
    print()
    
    # 9. Print final summary
    print("ANALYSIS COMPLETE!")
    print("=" * 18)
    print("Key findings emphasize gender similarities in mathematical cognition.")
    print("Individual differences exceed group-level patterns.")
    print("Cultural context has been integrated throughout the analysis.")
    print("Results support evidence-based educational approaches.")
    print()
    print("⚠️  ETHICAL REMINDER:")
    print("These findings should never be used to justify discrimination")
    print("or reinforce stereotypes. Focus on individual potential.")
    print()
    print(f"All results and visualizations saved to: {results_dir}")

if __name__ == '__main__':
    main()