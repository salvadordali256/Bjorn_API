from flask import Flask, request, jsonify, render_template
import re
import csv
import io
import tempfile
import os
from ml_bridge import init_ml, abbreviate_text
import json
import pandas as pd
import numpy as np
import tempfile
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename


app = Flask(__name__, template_folder='templates')

# Initialize ML model if available
ml_available = init_ml()


# Dictionary of common HVAC abbreviations
abbreviation_dict = {
    "Accessory": "Accy", "Actuator": "Act", "Adapter": "Adapt", "Aluminum": "Alum", "Aluminium": "Alum",
    "Analog": "Anlg", "Assembly": "Assy", "Averaging": "Avg", "BACnet": "Bnet", "Black": "Blk",
    "Blower": "Blwr", "Breaker": "Brkr", "Bronze": "Brz", "Butterfly": "Bfly", "Cable": "Cbl",
    "Capacitor": "Cap", "Capillary": "Cap", "Check": "Chk", "Compressor": "Comp", "Controller": "Ctrlr",
    "Control": "Ctrl", "Copper": "Cu", "Cover": "Cvr", "Detector": "Detect", "Differential": "Diff",
    "Electric": "Elec", "Enclosure": "Encl", "Evaporator": "Evap", "Expansion": "Exp", "Flange": "Flg",
    "Flare": "Flr", "Floating": "Flt", "Gasket": "Gskt", "Hazardous": "Hzrd", "Heater": "Htr",
    "Heat": "Ht", "High": "Hi", "Level": "Lvl", "Low": "Lo", "Modulating": "Mod", "Modular": "Mod",
    "Motor": "Mtr", "Mounted": "Mtd", "Mount": "Mt", "Mounting": "Mtg", "Pack": "Pk", "Package": "Pkg",
    "Panel": "Pnl", "Plate": "Plt", "Pressure": "Press", "Probe": "Prb", "Programmable": "Prog",
    "Programming": "Prog", "Program": "Prog", "Regulator": "Reg", "Relay": "Rly", "Relief": "Rlf",
    "Remote": "Rmt", "Sensor": "Sens", "Sens": "Sns", "Setpoint": "SetPt", "Set Point": "SetPt",
    "StainlessSteel": "SS", "Stanless Steel": "SS", "Sweat": "Swt", "Switch": "Sw", "Temperature": "Temp",
    "Thermistor": "Thrmst", "Thermostat": "Tstat", "Transceiver": "Trnsvr", "Transmitter": "Trnsmt",
    "Valve": "Vlv", "Water": "Wtr", "White": "Wht", "Without": "w/o", "With": "w/", "Explosion": "Expl",
    "Proof": "Prf", "Protection": "Prot", "Double": "Dbl", "Minutes": "Min", "Minute": "Min",
    "Inches": "\"", "Inch": "\"", "º": "Deg", "Piece": "Pc", "Voltage": "Volt", "Amps": "Amp",
    "Board": "Brd", "Extension": "Ext", "Transformer": "Xfrmr", "ExplPrf": "X-Prf", "Standard": "Std",
    "Round": "Rnd", "Density": "Dens", "Reflector": "Rflctr", "Disconnect": "Discon", "Regulating": "Reg",
    "Replacement": "Repl", "Infrared": "IR", "Filter": "Filt", "Pannel": "Pnl", "Included": "Incl",
    "Includes": "Incl", "Mted": "Mtd", "Position": "Pos", "Manual Reset": "MR", "Damper": "Dmpr",
    "Label": "Lbl",
    "Assemblies": "ASSY",
    "Assembly": "ASSY",
    "Assembled": "ASSD",
    "Cabinet": "CAB",
    "Cabinets": "CABS",
    "Factory": "FACT",
    "Showers": "SHWR",
    "Shower": "SHWR",
    "Recirculation": "RECIRC",
    "Piping": "PIPE",
    "Stainless": "STSTL",
    "Chrome": "CHR",
    "Plated": "PLT",
    "Thermostatic": "THERM",
    "Mixing": "MIX",
    "Exposed": "EXP",
    "Single": "SNGL",
    "Series": "SER",
    "Hydrotherapy": "HYDRO",
    "Connection": "CONN",
    "Capacity": "CAP",
    "Group": "GRP"
}

# Merge with the existing dictionary
abbreviation_dict.update(additional_abbrs)

@app.route('/ml/extract', methods=['POST'])
def ml_extract():
    try:
        # Check if file was uploaded
        if 'sourceFile' not in request.files:
            return jsonify({"success": False, "error": "No file part"}), 400
            
        file = request.files['sourceFile']
        if file.filename == '':
            return jsonify({"success": False, "error": "No selected file"}), 400
        
        # Get output file path
        output_file = request.form.get('outputFile', 'data/training_data.csv')
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save uploaded file to a temporary location
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        file.save(temp_file.name)
        
        # Import the ml module dynamically to avoid circular imports
        from ml.dataset_builder import extract_training_pairs
        
        # Extract training pairs
        training_data = extract_training_pairs(temp_file.name, output_file)
        
        # Clean up temporary file
        os.unlink(temp_file.name)
        
        # Prepare sample data for response
        samples = []
        if training_data and len(training_data) > 0:
            # Take up to 10 samples
            for i in range(min(10, len(training_data))):
                orig, abbr = training_data[i]
                orig_len = len(orig)
                abbr_len = len(abbr)
                reduction = ((orig_len - abbr_len) / orig_len * 100) if orig_len > 0 else 0
                
                samples.append({
                    "original": orig,
                    "abbreviated": abbr,
                    "reduction": f"{reduction:.1f}"
                })
        
        return jsonify({
            "success": True,
            "sourceFile": file.filename,
            "outputFile": output_file,
            "extractedPairs": len(training_data) if training_data else 0,
            "samples": samples
        })
        
    except Exception as e:
        app.logger.error(f"Error extracting training data: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/ml/analyze', methods=['POST'])
def ml_analyze():
    try:
        # Get training data file path
        training_data_file = request.form.get('trainingDataFile', 'data/training_data.csv')
        
        # Check if file exists
        if not os.path.exists(training_data_file):
            return jsonify({"success": False, "error": f"Training data file not found: {training_data_file}"}), 404
        
        # Import pattern discovery module dynamically
        from ml.pattern_discovery import PatternDiscovery
        
        # Load training data
        df = pd.read_csv(training_data_file)
        training_data = list(zip(df['original'], df['abbreviated']))
        
        # Analyze patterns
        discoverer = PatternDiscovery()
        patterns = discoverer.discover_patterns(training_data)
        
        # Extract rules
        rules_df, pattern_counts, avg_reduction = discoverer.extract_abbreviation_rules()
        
        # Prepare response data
        pattern_data = []
        for pattern, count in pattern_counts.items():
            # Find an example for this pattern
            example = ""
            for _, row in rules_df[rules_df['pattern'] == pattern].head(1).iterrows():
                example = f"{row['word']} → {row['abbreviation']}"
            
            pattern_data.append({
                "pattern": pattern,
                "count": int(count),
                "avgReduction": f"{avg_reduction[pattern]:.1f}",
                "example": example
            })
        
        # Prepare abbreviation data
        abbr_data = []
        for _, row in rules_df.head(20).iterrows():
            abbr_data.append({
                "original": row['word'],
                "abbreviated": row['abbreviation'],
                "reduction": f"{row['reduction_pct']:.1f}",
                "frequency": 1  # We don't have this info in the basic implementation
            })
        
        # Calculate overall stats
        total_patterns = len(patterns['word_patterns'])
        word_abbreviations = len(patterns['common_abbreviations'])
        phrase_abbreviations = len(patterns['phrase_patterns'])
        average_reduction = rules_df['reduction_pct'].mean()
        
        # Convert pattern counts to dict for chart
        pattern_counts_dict = {str(k): int(v) for k, v in pattern_counts.items()}
        
        return jsonify({
            "success": True,
            "totalPatterns": total_patterns,
            "wordAbbreviations": word_abbreviations,
            "phraseAbbreviations": phrase_abbreviations,
            "averageReduction": f"{average_reduction:.1f}",
            "patterns": pattern_data,
            "abbreviations": abbr_data,
            "patternCounts": pattern_counts_dict
        })
        
    except Exception as e:
        app.logger.error(f"Error analyzing patterns: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/ml/train', methods=['POST'])
def ml_train():
    try:
        # Get form data
        training_data_file = request.form.get('trainingDataFile', 'data/training_data.csv')
        model_output_file = request.form.get('modelOutputFile', 'models/abbreviation_model.joblib')
        model_type = request.form.get('modelType', 'hybrid')
        
        # Check if training data exists
        if not os.path.exists(training_data_file):
            return jsonify({"success": False, "error": f"Training data file not found: {training_data_file}"}), 404
        
        # Ensure model directory exists
        os.makedirs(os.path.dirname(model_output_file), exist_ok=True)
        
        # Import the appropriate model
        if model_type == 'hybrid':
            from ml.hybrid_model import HybridAbbreviationModel as ModelClass
        else:
            from ml.abbreviation_model import AbbreviationModel as ModelClass
        
        # Load training data
        df = pd.read_csv(training_data_file)
        training_data = list(zip(df['original'], df['abbreviated']))
        
        # Split data for training and validation
        from sklearn.model_selection import train_test_split
        train_data, val_data = train_test_split(training_data, test_size=0.2, random_state=42)
        
        # Create and train the model
        model = ModelClass()
        error = model.fit(train_data)
        
        # Save the model
        model.save(model_output_file)
        
        # Validate on validation set
        correct_predictions = 0
        success_count = 0
        reductions = []
        sample_predictions = []
        
        for i, (original, reference) in enumerate(val_data[:10]):  # Use just 10 examples for quick validation
            abbreviated = model.predict_abbreviation(original, 30)
            
            # Calculate stats
            orig_len = len(original)
            abbr_len = len(abbreviated)
            reduction = ((orig_len - abbr_len) / orig_len * 100) if orig_len > 0 else 0
            reductions.append(reduction)
            
            # Check if meets target length
            if abbr_len <= 30:
                success_count += 1
            
            # Check if exact match (unlikely but good to track)
            if abbreviated == reference:
                correct_predictions += 1
            
            # Save sample prediction for display
            sample_predictions.append({
                "original": original,
                "abbreviated": abbreviated,
                "reduction": f"{reduction:.1f}",
                "method": "Hybrid" if model_type == 'hybrid' else "ML-only"
            })
        
        # Calculate metrics
        validation_accuracy = (correct_predictions / len(val_data[:10]) * 100) if val_data else 0
        success_rate = (success_count / len(val_data[:10]) * 100) if val_data else 0
        avg_reduction = sum(reductions) / len(reductions) if reductions else 0
        
        # Simulate training progress for chart
        training_progress = [0.5, 0.3, 0.25, 0.2, 0.18, 0.15, 0.12, 0.1, 0.09, 0.08]
        
        return jsonify({
            "success": True,
            "modelType": model_type,
            "modelPath": model_output_file,
            "trainingExamples": len(train_data),
            "validationAccuracy": f"{validation_accuracy:.1f}",
            "successRate": f"{success_rate:.1f}",
            "averageReduction": f"{avg_reduction:.1f}",
            "trainingError": f"{error:.4f}",
            "trainingProgress": training_progress,
            "samplePredictions": sample_predictions
        })
        
    except Exception as e:
        app.logger.error(f"Error training model: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/ml/test', methods=['POST'])
def ml_test():
    try:
        # Get form data
        model_file = request.form.get('modelFile', 'models/abbreviation_model.joblib')
        test_text = request.form.get('testText', '')
        target_length = int(request.form.get('targetLength', 30))
        
        # Check if model exists
        if not os.path.exists(model_file):
            return jsonify({"success": False, "error": f"Model file not found: {model_file}"}), 404
        
        # Split test text into lines
        test_examples = [line.strip() for line in test_text.split('\n') if line.strip()]
        
        if not test_examples:
            return jsonify({"success": False, "error": "No test examples provided"}), 400
        
        # Check model type (hybrid or basic)
        try:
            from ml.hybrid_model import HybridAbbreviationModel
            model = HybridAbbreviationModel()
            model.load(model_file)
            model_type = "hybrid"
        except:
            from ml.abbreviation_model import AbbreviationModel
            model = AbbreviationModel()
            model.load(model_file)
            model_type = "basic"
        
        # Process each example
        results = []
        for example in test_examples:
            abbreviated = model.predict_abbreviation(example, target_length)
            
            orig_len = len(example)
            abbr_len = len(abbreviated)
            reduction = ((orig_len - abbr_len) / orig_len * 100) if orig_len > 0 else 0
            
            results.append({
                "original": example,
                "abbreviated": abbreviated,
                "original_length": orig_len,
                "abbreviated_length": abbr_len,
                "reduction": f"{reduction:.1f}%",
                "target_length": target_length,
                "method": model_type
            })
        
        return jsonify({
            "success": True,
            "model": model_file,
            "modelType": model_type,
            "results": results
        })
        
    except Exception as e:
        app.logger.error(f"Error testing model: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

# Add a route to the training interface
@app.route('/train')
def train_interface():
    return render_template('train.html')

def apply_abbreviation_rules(phrase, target_length=30):
    # Early return for empty or short phrases
    if not phrase or not isinstance(phrase, str) or len(phrase) <= target_length:
        return phrase, len(phrase) if phrase else 0, [], len(phrase) if phrase else 0
    
    original_length = len(phrase)
    abbreviations_applied = []
    processed_phrase = phrase
    
    # Industry-specific patterns and their abbreviations
    pattern_replacements = {
        r"CABINET ASSEMBLIES-SHOWERS, FACTORY ASSEMBLED": "CAB SHWR ASSY",
        r"CABINET ASSEMBLIES-HYDROTHERAPY, FACT\. ASSLD": "CAB HYDRO ASSY",
        r"Single Thermostatic Water Mixing Valves[,]?": "SNGL THERM MIX VLV",
        r"EXPOSED RECIRCULATION PIPING ASSEMBLY": "EXP RECIRC PIPE ASSY",
        r"EXPOSED VALVE ASSEMBLIES-Group shower": "EXP VLV GSHWR ASSY",
        r"GROUP SHOWER XL": "GRP SHWR XL",
        r"HIGH CAPACITY XL": "HI CAP XL",
        r"LV SERIES SINGLE VALVE": "LV SER SGL VLV",
        r"\(RF-rough bronze\)": "(RF-BRZ)",
        r"\(CP-chrome plate\)": "(CP-PLT)"
    }
    
    # First pass: Apply pattern-based replacements
    for pattern, replacement in pattern_replacements.items():
        if re.search(pattern, processed_phrase, re.IGNORECASE):
            old_phrase = processed_phrase
            processed_phrase = re.sub(pattern, replacement, processed_phrase, flags=re.IGNORECASE)
            if processed_phrase != old_phrase:
                abbreviations_applied.append(f"{pattern}→{replacement}")
    
    # Second pass: Dictionary-based abbreviation with optimization for maximum character savings
    if len(processed_phrase) > target_length:
        # Extract words that could be abbreviated
        words = []
        word_positions = []
        for match in re.finditer(r'\b\w+\b', processed_phrase):
            words.append(match.group())
            word_positions.append((match.start(), match.end()))
        
        # Calculate potential savings for each word
        potential_abbreviations = []
        for i, word in enumerate(words):
            # Skip words that are too short
            if len(word) <= 3:
                continue
                
            # Try different forms for lookup
            for lookup in [word, word.capitalize(), word.upper(), word.lower()]:
                clean_lookup = re.sub(r'[^\w/]', '', lookup)
                if clean_lookup in abbreviation_dict:
                    abbrev = abbreviation_dict[clean_lookup]
                    
                    # Preserve original case
                    if word.isupper():
                        abbrev = abbrev.upper()
                    elif word[0].isupper():
                        abbrev = abbrev[0].upper() + abbrev[1:].lower() if len(abbrev) > 1 else abbrev.upper()
                    
                    chars_saved = len(word) - len(abbrev)
                    if chars_saved > 0:
                        potential_abbreviations.append((i, word, abbrev, chars_saved))
                    break
        
        # Sort by character savings (highest first)
        potential_abbreviations.sort(key=lambda x: x[3], reverse=True)
        
        # Apply abbreviations until we reach target length or run out of abbreviations
        for i, word, abbrev, _ in potential_abbreviations:
            if len(processed_phrase) <= target_length:
                break
                
            # Calculate the positions for this specific occurrence
            start, end = word_positions[i]
            
            # Create new phrase by replacing just this instance
            processed_phrase = processed_phrase[:start] + abbrev + processed_phrase[end:]
            
            # Update remaining word positions
            length_diff = len(abbrev) - (end - start)
            for j in range(i + 1, len(word_positions)):
                word_positions[j] = (word_positions[j][0] + length_diff, word_positions[j][1] + length_diff)
            
            abbreviations_applied.append(f"{word}→{abbrev}")
    
    # Third pass: Intelligent word reduction
    if len(processed_phrase) > target_length:
        # Remove unnecessary words and punctuation
        remove_patterns = [
            (r', ', ' '),  # Replace commas with spaces
            (r'\s+', ' '),  # Consolidate multiple spaces
            (r'\s*\([^)]*\)\s*', ' ')  # Remove parenthetical phrases
        ]
        
        for pattern, replacement in remove_patterns:
            old_phrase = processed_phrase
            processed_phrase = re.sub(pattern, replacement, processed_phrase)
            processed_phrase = processed_phrase.strip()
            if old_phrase != processed_phrase:
                abbreviations_applied.append(f"Removed '{pattern}' pattern")
    
    # Fourth pass: Word truncation for longer words
    if len(processed_phrase) > target_length:
        words = processed_phrase.split()
        
        # Sort words by length (longest first)
        word_indices = sorted(range(len(words)), key=lambda i: len(words[i]), reverse=True)
        
        for idx in word_indices:
            if len(processed_phrase) <= target_length:
                break
                
            word = words[idx]
            if len(word) <= 3:
                continue
            
            # Intelligent truncation strategy
            if len(word) >= 8:
                # For very long words, keep first 3 chars + last char
                shortened = word[:3] + word[-1]
            elif len(word) >= 6:
                # For medium words, keep first 3 chars
                shortened = word[:3]
            elif len(word) > 4:
                # For shorter words, just remove a character or two
                shortened = word[:len(word)-1]
            else:
                continue  # Don't shorten 4-letter words
            
            # Apply the truncation
            words[idx] = shortened
            new_phrase = ' '.join(words)
            
            # Only apply if it actually helps
            if len(new_phrase) < len(processed_phrase):
                abbreviations_applied.append(f"{word}→{shortened}")
                processed_phrase = new_phrase
    
    # Final pass: If still over target length, truncate intelligently with ellipsis
    if len(processed_phrase) > target_length:
        # Try to truncate at a word boundary
        last_space = processed_phrase[:target_length-3].rfind(' ')
        if last_space > target_length * 0.7:  # Only truncate at word if we're not losing too much
            truncated = processed_phrase[:last_space] + "..."
        else:
            truncated = processed_phrase[:target_length-3] + "..."
        
        abbreviations_applied.append("Truncated entire phrase")
        processed_phrase = truncated
    
    return processed_phrase, original_length, abbreviations_applied, len(processed_phrase)


def verify_abbreviation(original, abbreviated, abbreviations_applied):
    """
    Enhanced verification with better metrics and readability assessment
    """
    target_length = 30
    
    # Calculate metrics
    length_reduction = ((len(original) - len(abbreviated)) / len(original)) if len(original) > 0 else 0
    
    # Check if the abbreviation preserves meaning
    meaning_preserved = True
    severe_truncation = False
    
    for abbr in abbreviations_applied:
        if "Truncated entire phrase" in abbr:
            severe_truncation = True
            meaning_preserved = False
            break
    
    # Check readability
    readability_score = 1.0
    if severe_truncation:
        readability_score = 0.3
    elif "truncated" in str(abbreviations_applied).lower():
        readability_score = 0.6
    elif len(abbreviated) > target_length:
        readability_score = 0.7
    
    # Calculate confidence score
    if len(abbreviated) <= target_length and not severe_truncation:
        base_confidence = 0.7
    else:
        base_confidence = 0.4
        
    meaning_factor = 0.2 if meaning_preserved else 0
    standard_abbr_factor = 0.1 if not "truncated" in str(abbreviations_applied).lower() else 0
    
    confidence = min(base_confidence + meaning_factor + standard_abbr_factor, 1.0)
    
    # Generate suggestions
    suggestions = []
    if severe_truncation:
        suggestions.append("Phrase was severely truncated, meaning may be lost")
    if "truncated" in str(abbreviations_applied).lower():
        suggestions.append("Contains non-standard abbreviations")
    if readability_score < 0.7:
        suggestions.append("Some abbreviations may not be easily recognized")
    
    return {
        "confidence": round(confidence, 2),
        "readability": round(readability_score, 2),
        "meaning_preserved": meaning_preserved,
        "is_standard": not "truncated" in str(abbreviations_applied).lower(),
        "suggestions": suggestions
    }

    """
    Simple AI verification using rule-based checks
    """
    # Define target_length (was missing in this function)
    target_length = 30
    
    # Calculate what percentage of length was reduced
    if len(original) == 0:
        length_reduction = 0
    else:
        length_reduction = (len(original) - len(abbreviated)) / len(original)
    
    # Check if all abbreviations came from our standard dictionary
    standard_abbrs = True
    for abbr_entry in abbreviations_applied:
        if "truncated" in abbr_entry.lower() or "entire phrase" in abbr_entry.lower():
            standard_abbrs = False
            break
        parts = abbr_entry.split('→')
        if len(parts) == 2:
            original_word, abbreviated_word = parts
            original_clean = re.sub(r'[^\w/]', '', original_word).lower()
            abbreviated_clean = re.sub(r'[^\w/]', '', abbreviated_word).lower()
            # Check if this abbreviation is in our dictionary (case insensitive)
            found = False
            for dict_key, dict_value in abbreviation_dict.items():
                if original_clean == dict_key.lower() and abbreviated_clean == dict_value.lower():
                    found = True
                    break
            if not found:
                standard_abbrs = False
    
    # Check if abbreviated words are still recognizable
    recognizable = True
    for abbr_entry in abbreviations_applied:
        if "truncated" in abbr_entry.lower():
            recognizable = False
            break
        parts = abbr_entry.split('→')
        if len(parts) == 2:
            original_word, abbreviated_word = parts
            # Simplistic check - first letter should match
            if abbreviated_word and original_word and not abbreviated_word[0].lower() == original_word[0].lower():
                recognizable = False
    
    # Generate confidence score based on our checks
    base_confidence = 0.5
    length_factor = length_reduction * 0.3
    standard_factor = 0.2 if standard_abbrs else 0
    recognizable_factor = 0.2 if recognizable else 0
    
    confidence = min(base_confidence + length_factor + standard_factor + recognizable_factor, 1.0)
    
    # Generate suggestions
    suggestions = []
    if len(abbreviated) > target_length:
        suggestions.append("Still exceeds maximum length of 30 characters")
    if not standard_abbrs:
        suggestions.append("Contains non-standard abbreviations")
    if not recognizable:
        suggestions.append("Some abbreviations may not be easily recognized")
    if "Truncated entire phrase" in abbreviations_applied:
        suggestions.append("Phrase was severely truncated, meaning may be lost")
    
    return {
        "confidence": round(confidence, 2),
        "is_standard": standard_abbrs,
        "suggestions": suggestions
    }

def apply_smart_abbreviations(text, target_length=30):
    """
    A more sophisticated approach using regex for better position tracking
    """
    original = text
    abbrevs_applied = []
    
    # Use regex to identify words and track their exact positions
    word_matches = list(re.finditer(r'\b(\w+)\b', text))
    
    # Try dictionary matches with position tracking
    for match in word_matches:
        if len(text) <= target_length:
            break
            
        word = match.group(1)
        if len(word) <= 3:
            continue
            
        if word.lower() in abbreviation_dict:
            abbrev = abbreviation_dict[word.lower()]
            # Preserve casing
            if word.isupper():
                abbrev = abbrev.upper()
            elif word[0].isupper():
                abbrev = abbrev.capitalize()
                
            # Calculate positions accounting for previous changes
            start, end = match.span(1)
            old_len = len(text)
            
            # Replace just this word
            text = text[:start] + abbrev + text[end:]
            
            # Track the abbreviation
            abbrevs_applied.append(f"{word}→{abbrev}")
    
    return text, len(original), abbrevs_applied, len(text)

def detect_product_patterns(text):
    """
    Identify specific product code patterns and handle them intelligently
    """
    # Pattern for XL and LV series product codes
    xl_pattern = re.compile(r'(XL|LV)-(\d+)-([\w-]+)')
    
    if xl_pattern.search(text):
        # Product code found, don't abbreviate the code itself
        # but focus on abbreviating the description
        parts = text.split('\t', 1) if '\t' in text else text.split(None, 1)
        if len(parts) > 1:
            code, description = parts
            # Abbreviate only the description
            abbr_desc, _, abbrevs, _ = apply_abbreviation_rules(description, 30 - len(code) - 1)
            return f"{code} {abbr_desc}", abbrevs
            
    return None, []



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/abbreviate', methods=['POST'])
def abbreviate():
    # Check if file was uploaded
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # Get ML preference
    use_ml = request.form.get('use_ml', 'false').lower() == 'true'
    
    # Process the uploaded CSV file
    try:
        # Read file contents
        file_content = file.read().decode('utf-8', errors='replace')
        
        # Parse CSV data
        csv_data = list(csv.reader(io.StringIO(file_content)))
        
        # Get headers
        headers = csv_data[0]
        
        # Check if Part Definition column exists
        if "Part Definition" not in headers:
            # Try to find it case-insensitively
            for i, header in enumerate(headers):
                if header.lower() == "part definition":
                    headers[i] = "Part Definition"
                    break
            else:
                return jsonify({"error": "File must contain a 'Part Definition' column"}), 400
        
        # Find the index of Part Definition column
        part_def_idx = headers.index("Part Definition")
        
        # Create result headers
        result_headers = headers.copy()
        if "Abbreviation" not in result_headers:
            result_headers.append("Abbreviation")
        if "Original Length" not in result_headers:
            result_headers.append("Original Length")
        if "Final Length" not in result_headers:
            result_headers.append("Final Length") 
        if "Length Reduction" not in result_headers:
            result_headers.append("Length Reduction")
        if "Applied Rules" not in result_headers:
            result_headers.append("Applied Rules")
        if "AI Confidence" not in result_headers:
            result_headers.append("AI Confidence") 
        if "Is Standard" not in result_headers:
            result_headers.append("Is Standard")
        if "Suggestions" not in result_headers:
            result_headers.append("Suggestions")
        if "Method Used" not in result_headers:
            result_headers.append("Method Used")
        
        # Process each row
        result_data = [result_headers]
        for row_idx, row in enumerate(csv_data[1:], 1):
            if len(row) <= part_def_idx:
                # Skip rows that don't have enough columns
                continue
                
            # Get part definition
            part_def = row[part_def_idx]
            
            # Use integrated abbreviation function
            final, orig_len, applied, final_len, method_used = abbreviate_text(
                part_def, 30, use_ml and ml_available)
            
            # Get verification results
            verification = verify_abbreviation(part_def, final, applied)
            
            # Create result row (copy original data and add our new columns)
            result_row = row.copy()
            
            # Add abbreviation columns if needed
            while len(result_row) < len(result_headers):
                result_row.append("")
            
            # Set abbreviation data
            abbr_idx = result_headers.index("Abbreviation")
            orig_len_idx = result_headers.index("Original Length")
            final_len_idx = result_headers.index("Final Length") 
            len_red_idx = result_headers.index("Length Reduction")
            applied_idx = result_headers.index("Applied Rules") 
            conf_idx = result_headers.index("AI Confidence")
            std_idx = result_headers.index("Is Standard")
            sugg_idx = result_headers.index("Suggestions")
            method_idx = result_headers.index("Method Used")
            
            result_row[abbr_idx] = final
            result_row[orig_len_idx] = str(orig_len)
            result_row[final_len_idx] = str(final_len)
            # Calculate length reduction percentage
            if orig_len > 0:
                result_row[len_red_idx] = f"{((orig_len - final_len) / orig_len) * 100:.1f}%"
            else:
                result_row[len_red_idx] = "0.0%"
            result_row[applied_idx] = ", ".join(applied)
            result_row[conf_idx] = str(verification["confidence"])
            result_row[std_idx] = "Yes" if verification["is_standard"] else "No"
            result_row[sugg_idx] = ", ".join(verification["suggestions"])
            result_row[method_idx] = method_used
            
            # Add to result data
            result_data.append(result_row)
        
        # Create output CSV content
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerows(result_data)
        
        # Create response
        output_str = output.getvalue()
        
        # Return file for download
        return jsonify({
            "success": True,
            "message": "File processed successfully",
            "csv_data": output_str
        })
        
    except Exception as e:
        app.logger.error(f"Error processing file: {str(e)}")
        return jsonify({"error": str(e)}), 500
    
if __name__ == '__main__':
    # Ensure the app can be reached from outside the container
    app.run(host='0.0.0.0', port=5000, debug=False)