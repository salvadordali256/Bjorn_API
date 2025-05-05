#!/usr/bin/env python3
import argparse
import os
import sys
import pandas as pd
from hybrid_model import HybridAbbreviationModel
from dataset_builder import extract_training_pairs, analyze_abbreviations
from pattern_discovery import PatternDiscovery, analyze_training_data

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='ML-based Part Description Abbreviator')
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Extract training data command
    extract_parser = subparsers.add_parser('extract', help='Extract training data from existing abbreviations')
    extract_parser.add_argument('source', help='Source file with original and abbreviated descriptions')
    extract_parser.add_argument('-o', '--output', default='training_data.csv', help='Output CSV file for training data')
    
    # Analyze patterns command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze abbreviation patterns in training data')
    analyze_parser.add_argument('training_data', help='CSV file with training data')
    
    # Train model command
    train_parser = subparsers.add_parser('train', help='Train a new abbreviation model')
    train_parser.add_argument('training_data', help='CSV file with training data')
    train_parser.add_argument('-o', '--output', default='models/abbreviation_model.joblib', help='Output path for model')
    
    # Test model command
    test_parser = subparsers.add_parser('test', help='Test a trained model on examples')
    test_parser.add_argument('model', help='Path to trained model file')
    test_parser.add_argument('text', nargs='*', help='Example texts to abbreviate')
    test_parser.add_argument('-f', '--file', help='File with texts to abbreviate (one per line)')
    test_parser.add_argument('-l', '--length', type=int, default=30, help='Target abbreviation length')
    
    # Abbreviate file command
    abbr_parser = subparsers.add_parser('abbreviate', help='Abbreviate descriptions in a CSV file')
    abbr_parser.add_argument('input_file', help='Input CSV file')
    abbr_parser.add_argument('model', help='Path to trained model file')
    abbr_parser.add_argument('-o', '--output', help='Output CSV file')
    abbr_parser.add_argument('-c', '--column', default='Part Definition', help='Column with descriptions to abbreviate')
    abbr_parser.add_argument('-l', '--length', type=int, default=30, help='Target abbreviation length')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Handle commands
    if args.command == 'extract':
        print(f"Extracting training data from {args.source} to {args.output}")
        training_data = extract_training_pairs(args.source, args.output)
        if training_data:
            print(f"Successfully extracted {len(training_data)} training pairs")
            
            # Optional: perform quick analysis
            print("\nAnalyzing abbreviation patterns...")
            common_abbreviations, patterns, _ = analyze_abbreviations(training_data)
            
            print(f"Found {len(common_abbreviations)} common abbreviations")
            print("\nTop 10 common abbreviations:")
            for word, abbr in list(common_abbreviations.items())[:10]:
                print(f"  {word} → {abbr}")
                
    elif args.command == 'analyze':
        print(f"Analyzing abbreviation patterns in {args.training_data}")
        discoverer, rules_df, _ = analyze_training_data(args.training_data)
        
        # Print additional statistics
        print("\nTop words by reduction percentage:")
        top_by_reduction = rules_df.sort_values('reduction_pct', ascending=False).head(10)
        for _, row in top_by_reduction.iterrows():
            print(f"  {row['word']} → {row['abbreviation']} ({row['reduction_pct']:.1f}% reduction)")
        
    elif args.command == 'train':
        print(f"Training model on {args.training_data}")
        model = train_and_save_model(args.training_data, args.output)
        
        # Print some examples of model prediction
        print("\nTesting model on some examples:")
        examples = [
            "CABINET ASSEMBLIES-SHOWERS, FACTORY ASSEMBLED (RF-rough bronze)",
            "EXPOSED RECIRCULATION PIPING ASSEMBLY, ROUGH BRONZE",
            "Single Thermostatic Water Mixing Valves, CHROME PLATED"
        ]
        
        for example in examples:
            abbreviated = model.predict_abbreviation(example)
            print(f"\nOriginal ({len(example)}): {example}")
            print(f"Abbreviated ({len(abbreviated)}): {abbreviated}")
        
    elif args.command == 'test':
        if not os.path.exists(args.model):
            print(f"Error: Model file {args.model} not found")
            sys.exit(1)
            
        print(f"Loading model from {args.model}")
        model = HybridAbbreviationModel()
        if not model.load(args.model):
            print("Error loading model")
            sys.exit(1)
        
        # Get texts to abbreviate
        texts = args.text
        
        if args.file and os.path.exists(args.file):
            with open(args.file, 'r', encoding='utf-8') as f:
                texts.extend([line.strip() for line in f if line.strip()])
        
        if not texts:
            print("No texts provided to abbreviate")
            sys.exit(1)
            
        print(f"\nAbbreviating {len(texts)} texts to target length {args.length}:")
        for text in texts:
            abbreviated = model.predict_abbreviation(text, args.length)
            original_len = len(text)
            abbreviated_len = len(abbreviated)
            reduction = ((original_len - abbreviated_len) / original_len) * 100 if original_len > 0 else 0
            
            print(f"\nOriginal ({original_len}): {text}")
            print(f"Abbreviated ({abbreviated_len}): {abbreviated}")
            print(f"Reduction: {reduction:.1f}%")
            
    elif args.command == 'abbreviate':
        if not os.path.exists(args.input_file):
            print(f"Error: Input file {args.input_file} not found")
            sys.exit(1)
            
        if not os.path.exists(args.model):
            print(f"Error: Model file {args.model} not found")
            sys.exit(1)
            
        # Load model
        print(f"Loading model from {args.model}")
        model = HybridAbbreviationModel()
        if not model.load(args.model):
            print("Error loading model")
            sys.exit(1)
        
        # Load CSV
        print(f"Processing file {args.input_file}")
        try:
            df = pd.read_csv(args.input_file)
        except Exception as e:
            print(f"Error reading CSV: {str(e)}")
            sys.exit(1)
            
        # Check if column exists
        if args.column not in df.columns:
            # Try case-insensitive search
            col_found = False
            for col in df.columns:
                if col.lower() == args.column.lower():
                    args.column = col
                    col_found = True
                    break
                    
            if not col_found:
                print(f"Error: Column '{args.column}' not found in input file")
                sys.exit(1)
        
        # Add results columns
        df['Abbreviated'] = df[args.column].apply(lambda x: model.predict_abbreviation(str(x), args.length) if pd.notna(x) else '')
        df['Original_Length'] = df[args.column].apply(lambda x: len(str(x)) if pd.notna(x) else 0)
        df['Abbreviated_Length'] = df['Abbreviated'].apply(len)
        df['Reduction_Pct'] = ((df['Original_Length'] - df['Abbreviated_Length']) / df['Original_Length'] * 100).fillna(0)
        
        # Save results
        output_file = args.output if args.output else args.input_file.rsplit('.', 1)[0] + '_abbreviated.csv'
        df.to_csv(output_file, index=False)
        
        print(f"Results saved to {output_file}")
        print(f"Processed {len(df)} rows")
        
        # Print summary
        avg_reduction = df['Reduction_Pct'].mean()
        success_rate = (df['Abbreviated_Length'] <= args.length).mean() * 100
        
        print(f"Average reduction: {avg_reduction:.1f}%")
        print(f"Success rate (<=30 chars): {success_rate:.1f}%")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()