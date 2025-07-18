#!/usr/bin/env python3
"""
MCP-fMRI Command Line Interface
CLI tools for preprocessing and analysis
"""

import argparse
import sys
from pathlib import Path
import logging
from typing import Optional

from .preprocessing import fMRIPreprocessor
from .analysis import GenderSimilarityAnalyzer, EthicalfMRIAnalysis
from .cultural import JapaneseCulturalContext
from .visualization import EthicalReportGenerator
from .utils import load_config, validate_data

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def preprocess_cli():
    """Command line interface for preprocessing."""
    parser = argparse.ArgumentParser(
        description='MCP-fMRI Preprocessing Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  mcp-fmri-preprocess --input /data/raw --output /data/processed
  mcp-fmri-preprocess --input /data/raw --output /data/processed --cultural-context japanese
  mcp-fmri-preprocess --config preprocessing_config.yaml
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        help='Input directory containing raw fMRI data'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output directory for preprocessed data'
    )
    
    parser.add_argument(
        '--participants', '-p',
        nargs='+',
        help='Specific participant IDs to process (e.g., JP001 JP002)'
    )
    
    parser.add_argument(
        '--cultural-context', '-c',
        type=str,
        default='japanese',
        choices=['japanese'],
        help='Cultural context for analysis (default: japanese)'
    )
    
    parser.add_argument(
        '--ethical-guidelines',
        action='store_true',
        default=True,
        help='Enable ethical preprocessing guidelines'
    )
    
    parser.add_argument(
        '--motion-threshold',
        type=float,
        default=3.0,
        help='Motion threshold in mm (default: 3.0)'
    )
    
    parser.add_argument(
        '--smoothing-fwhm',
        type=float,
        default=8.0,
        help='Smoothing FWHM in mm (default: 8.0)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Configuration file path (YAML format)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration if provided
    config = {}
    if args.config:
        config = load_config(args.config)
        logger.info(f"Configuration loaded from {args.config}")
    
    # Use command line arguments or config file values
    input_dir = args.input or config.get('input_dir')
    output_dir = args.output or config.get('output_dir')
    cultural_context = args.cultural_context or config.get('cultural_context', 'japanese')
    
    if not input_dir or not output_dir:
        logger.error("Input and output directories are required")
        parser.print_help()
        sys.exit(1)
    
    # Initialize preprocessor
    preprocessor = fMRIPreprocessor(
        raw_data_dir=input_dir,
        output_dir=output_dir,
        ethical_guidelines=args.ethical_guidelines,
        cultural_context=cultural_context
    )
    
    # Update parameters
    preprocessor.params['motion_threshold'] = args.motion_threshold or config.get('motion_threshold', 3.0)
    preprocessor.params['smoothing_fwhm'] = args.smoothing_fwhm or config.get('smoothing_fwhm', 8.0)
    
    # Determine participant list
    if args.participants:
        participant_list = args.participants
        logger.info(f"Processing specified participants: {participant_list}")
    else:
        # Auto-generate participant list or load from config
        participant_list = config.get('participants', [f"JP{i:03d}" for i in range(1, 157)])
        logger.info(f"Processing {len(participant_list)} participants")
    
    try:
        # Run preprocessing
        logger.info("Starting preprocessing pipeline...")
        results = preprocessor.preprocess_batch(participant_list)
        
        # Generate QC report
        preprocessor.generate_qc_report()
        
        # Summary
        n_success = len(results[results['status'] == 'success'])
        n_pass_qc = len(results[results['passes_qc'] == True])
        
        logger.info(f"Preprocessing completed successfully!")
        logger.info(f"Processed: {n_success}/{len(participant_list)} participants")
        logger.info(f"Passed QC: {n_pass_qc}/{len(participant_list)} participants")
        logger.info(f"Results saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        sys.exit(1)

def analyze_cli():
    """Command line interface for analysis."""
    parser = argparse.ArgumentParser(
        description='MCP-fMRI Gender Similarity Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  mcp-fmri-analyze --data /data/processed --output /results
  mcp-fmri-analyze --data /data/processed --cultural-context japanese --similarity-threshold 0.8
  mcp-fmri-analyze --config analysis_config.yaml
        """
    )
    
    parser.add_argument(
        '--data', '-d',
        type=str,
        help='Directory containing preprocessed data'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output directory for analysis results'
    )
    
    parser.add_argument(
        '--cultural-context', '-c',
        type=str,
        default='japanese',
        choices=['japanese'],
        help='Cultural context for analysis (default: japanese)'
    )
    
    parser.add_argument(
        '--similarity-threshold',
        type=float,
        default=0.8,
        help='Similarity threshold for analysis (default: 0.8)'
    )
    
    parser.add_argument(
        '--bias-detection',
        action='store_true',
        default=True,
        help='Enable bias detection analysis'
    )
    
    parser.add_argument(
        '--generate-plots',
        action='store_true',
        default=True,
        help='Generate visualization plots'
    )
    
    parser.add_argument(
        '--interactive-dashboard',
        action='store_true',
        help='Generate interactive dashboard'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Configuration file path (YAML format)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration if provided
    config = {}
    if args.config:
        config = load_config(args.config)
        logger.info(f"Configuration loaded from {args.config}")
    
    # Use command line arguments or config file values
    data_dir = args.data or config.get('data_dir')
    output_dir = args.output or config.get('output_dir', './results')
    cultural_context = args.cultural_context or config.get('cultural_context', 'japanese')
    
    if not data_dir:
        logger.error("Data directory is required")
        parser.print_help()
        sys.exit(1)
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize cultural context
        if cultural_context == 'japanese':
            context = JapaneseCulturalContext()
        else:
            context = None
        
        # Initialize analyzer
        analyzer = GenderSimilarityAnalyzer(
            ethical_guidelines=True,
            similarity_threshold=args.similarity_threshold or config.get('similarity_threshold', 0.8),
            cultural_context=cultural_context,
            bias_detection=args.bias_detection
        )
        
        logger.info("Starting similarity analysis...")
        
        # Load and validate data
        data_dict = analyzer.load_preprocessed_data(data_dir)
        
        if context:
            validation_results = context.validate_sample(data_dict['demographics'])
            if not validation_results['sample_representative']:
                logger.warning("Sample may not be fully representative - consider cultural factors")
        
        # Run analysis
        similarities, bias_results = analyzer.analyze_similarities(data_dict)
        
        # Cultural interpretation
        if context:
            cultural_interpretation = context.interpret_results(similarities, bias_results)
            logger.info("Cultural interpretation completed")
        else:
            cultural_interpretation = {}
        
        # Generate report
        report_generator = EthicalReportGenerator(cultural_context=cultural_context)
        analysis_results = {
            'similarities': similarities,
            'bias_detection': bias_results,
            'cultural_interpretation': cultural_interpretation,
            'cultural_context': context
        }
        
        # Generate comprehensive report
        if args.generate_plots:
            report = report_generator.generate_similarity_report(
                analysis_results, 
                output_dir=output_dir
            )
        else:
            report = {'text_report': analyzer.generate_report()}
        
        # Save text report
        report_path = Path(output_dir) / 'similarity_analysis_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report['text_report'])
        
        # Save detailed results
        import json
        results_path = Path(output_dir) / 'analysis_results.json'
        
        # Convert numpy arrays to lists for JSON serialization
        json_safe_results = {}
        for key, value in analysis_results.items():
            if key == 'cultural_context':
                continue  # Skip non-serializable context object
            elif isinstance(value, dict):
                json_safe_results[key] = {k: float(v) if isinstance(v, (int, float)) else v 
                                        for k, v in value.items()}
            else:
                json_safe_results[key] = value
        
        with open(results_path, 'w') as f:
            json.dump(json_safe_results, f, indent=2)
        
        # Generate interactive dashboard if requested
        if args.interactive_dashboard and 'figures' in report:
            dashboard_path = Path(output_dir) / 'interactive_dashboard.html'
            if 'interactive_dashboard' in report['figures']:
                report['figures']['interactive_dashboard'].write_html(str(dashboard_path))
                logger.info(f"Interactive dashboard saved to {dashboard_path}")
        
        # Print summary
        logger.info("Analysis completed successfully!")
        logger.info(f"Overall similarity index: {similarities['overall_similarity_index']:.3f}")
        logger.info(f"Individual:Group ratio: {similarities['individual_to_group_ratio']:.2f}:1")
        logger.info(f"Mean effect size: {similarities['mean_cohens_d']:.3f}")
        
        if bias_results:
            logger.info(f"Classification accuracy: {bias_results['classification_accuracy']:.3f}")
            logger.info(f"Bias risk: {bias_results['bias_risk']}")
        
        logger.info(f"Results saved to: {output_dir}")
        
        # Print ethical reminder
        print("\n" + "="*60)
        print("ETHICAL REMINDER:")
        print("This analysis emphasizes gender similarities over differences.")
        print("Results should be interpreted with cultural context and should")
        print("never be used to justify discrimination or stereotypes.")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)

def main():
    """Main entry point for CLI."""
    if len(sys.argv) < 2:
        print("MCP-fMRI: Ethical Analysis of Mathematical Abilities Using Generative AI")
        print("")
        print("Available commands:")
        print("  mcp-fmri-preprocess  - Run fMRI preprocessing pipeline")
        print("  mcp-fmri-analyze     - Run gender similarity analysis")
        print("")
        print("Use --help with any command for detailed usage information.")
        sys.exit(1)

if __name__ == '__main__':
    main()