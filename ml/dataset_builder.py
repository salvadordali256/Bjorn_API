import pandas as pd
import csv
import os
import re

def extract_training_pairs(file_path, output_path):
    """Extract pairs of original and abbreviated text from existing data"""
    training_data = []
    
    # Read the file as tab-separated 
    df = pd.read_csv(file_path, sep='\t', header=None, error_bad_lines=False)
    
    # We need at least 3 columns: original part number, description, and shortened description
    if df.shape[1] >= 3:
        for _, row in df.iterrows():
            # Extract the original description and abbreviated version
            original = str(row[1]) if pd.notna(row[1]) else ""
            abbreviated = str(row[2]) if pd.notna(row[2]) else ""
            
            # Only include valid pairs where abbreviation is shorter
            if original and abbreviated and len(abbreviated) < len(original):
                training_data.append((original, abbreviated))
    
    # If training_data is not empty, save as CSV
    if training_data:
        train_df = pd.DataFrame(training_data, columns=['original', 'abbreviated'])
        train_df.to_csv(output_path, index=False)
        print(f"Extracted {len(training_data)} training pairs. Saved to {output_path}")
    else:
        print("No valid training pairs found in the file.")
    
    return training_data

def analyze_abbreviations(training_data):
    """Analyze abbreviation patterns to identify transformation rules"""
    word_level_patterns = {}
    character_reduction = {}
    
    for original, abbreviated in training_data:
        # Split into words
        orig_words = original.split()
        abbr_words = abbreviated.split()
        
        # Analyze word-level transformations
        if len(orig_words) == len(abbr_words):
            for i, (orig_word, abbr_word) in enumerate(zip(orig_words, abbr_words)):
                if orig_word != abbr_word:
                    # Record the transformation
                    if orig_word not in word_level_patterns:
                        word_level_patterns[orig_word] = {}
                    
                    if abbr_word in word_level_patterns[orig_word]:
                        word_level_patterns[orig_word][abbr_word] += 1
                    else:
                        word_level_patterns[orig_word][abbr_word] = 1
                    
                    # Record character reduction
                    char_diff = len(orig_word) - len(abbr_word)
                    if char_diff > 0:
                        if orig_word not in character_reduction:
                            character_reduction[orig_word] = []
                        character_reduction[orig_word].append((abbr_word, char_diff))
    
    # Find the most common abbreviation for each word
    common_abbreviations = {}
    for word, abbrevs in word_level_patterns.items():
        if abbrevs:
            common_abbreviations[word] = max(abbrevs.items(), key=lambda x: x[1])[0]
    
    # Save results to files
    with open('common_abbreviations.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['original', 'abbreviated', 'frequency'])
        for word, abbrevs in word_level_patterns.items():
            for abbr, count in abbrevs.items():
                writer.writerow([word, abbr, count])
    
    print(f"Found {len(common_abbreviations)} common word abbreviations")
    return common_abbreviations, word_level_patterns, character_reduction

if __name__ == "__main__":
    # Path to your data file
    file_path = "paste-2.txt"
    output_path = "training_data.csv"
    
    # Extract training pairs
    training_data = extract_training_pairs(file_path, output_path)
    
    # Analyze abbreviation patterns
    if training_data:
        common_abbreviations, patterns, reductions = analyze_abbreviations(training_data)
        
        # Print some examples
        print("\nExamples of common abbreviations:")
        count = 0
        for word, abbr in common_abbreviations.items():
            print(f"{word} → {abbr}")
            count += 1
            if count >= 10:
                break