# hybrid_model.py
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import joblib
import re
import os
from advanced_features import AbbreviationFeatures

class HybridAbbreviationModel:
    """Hybrid model combining ML with rule-based abbreviation"""
    
    def __init__(self):
        # ML pipeline
        self.pipeline = Pipeline([
            ('char_features', CountVectorizer(analyzer='char', ngram_range=(1, 3))),
            ('text_features', AbbreviationFeatures()),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        
        # Dictionary for common word abbreviations
        self.word_abbreviations = {}
        
        # Dictionary for common phrase abbreviations
        self.phrase_abbreviations = {}
        
        # Status flag
        self.is_trained = False
    
    def fit(self, training_data):
        """Train the model on pairs of (original, abbreviated) texts"""
        # Extract original and abbreviated texts
        original_texts = [item[0] for item in training_data]
        abbreviated_texts = [item[1] for item in training_data]
        
        # Extract abbreviation dictionaries
        self._extract_abbreviation_patterns(training_data)
        
        # Calculate abbreviation ratios for ML training
        abbreviation_ratios = np.array([len(abbr)/len(orig) for orig, abbr in training_data])
        
        # Train test split for validation
        X_train, X_test, y_train, y_test = train_test_split(
            original_texts, abbreviation_ratios, test_size=0.2, random_state=42)
        
        # Train the pipeline
        self.pipeline.fit(X_train, y_train)
        
        # Calculate prediction error
        y_pred = self.pipeline.predict(X_test)
        mae = np.mean(np.abs(y_test - y_pred))
        
        self.is_trained = True
        
        print(f"Model trained on {len(X_train)} examples, validated on {len(X_test)} examples")
        print(f"Mean Absolute Error: {mae:.4f}")
        print(f"Extracted {len(self.word_abbreviations)} word abbreviations")
        print(f"Extracted {len(self.phrase_abbreviations)} phrase abbreviations")
        
        return mae
    
    def _extract_abbreviation_patterns(self, training_data):
        """Extract abbreviation patterns from training data"""
        # Word-level abbreviations
        word_patterns = {}
        
        # Phrase-level abbreviations (for frequent domain-specific phrases)
        phrase_patterns = {}
        
        for original, abbreviated in training_data:
            # Process exact phrase matches
            orig_phrase = original.strip()
            abbr_phrase = abbreviated.strip()
            
            if orig_phrase not in phrase_patterns:
                phrase_patterns[orig_phrase] = {}
            
            if abbr_phrase in phrase_patterns[orig_phrase]:
                phrase_patterns[orig_phrase][abbr_phrase] += 1
            else:
                phrase_patterns[orig_phrase][abbr_phrase] = 1
            
            # Process word by word
            orig_words = original.split()
            abbr_words = abbreviated.split()
            
            # Only if same number of words
            if len(orig_words) == len(abbr_words):
                for orig, abbr in zip(orig_words, abbr_words):
                    if orig != abbr and len(orig) > len(abbr):
                        if orig not in word_patterns:
                            word_patterns[orig] = {}
                        
                        if abbr in word_patterns[orig]:
                            word_patterns[orig][abbr] += 1
                        else:
                            word_patterns[orig][abbr] = 1
        
        # Extract most common abbreviations
        for word, abbrevs in word_patterns.items():
            if abbrevs:
                self.word_abbreviations[word] = max(abbrevs.items(), key=lambda x: x[1])[0]
        
        # Extract common phrase abbreviations
        for phrase, abbrevs in phrase_patterns.items():
            if abbrevs and len(phrase) > 5:  # Only consider non-trivial phrases
                self.phrase_abbreviations[phrase] = max(abbrevs.items(), key=lambda x: x[1])[0]
    
    def predict_abbreviation(self, text, target_length=30):
        """Apply hybrid abbreviation to a text"""
        if not self.is_trained:
            return text
            
        # If already short enough, return as is
        if len(text) <= target_length:
            return text
            
        # Check for exact phrase match
        if text in self.phrase_abbreviations:
            return self.phrase_abbreviations[text]
        
        # Try to apply our domain-specific phrase abbreviations
        for phrase, abbrev in sorted(self.phrase_abbreviations.items(), key=lambda x: len(x[0]), reverse=True):
            if phrase in text:
                text = text.replace(phrase, abbrev)
                # If success, check if we're done
                if len(text) <= target_length:
                    return text
        
        # Split into words and apply word-level abbreviations
        words = text.split()
        abbreviated_words = []
        
        for word in words:
            # Check if we have a known abbreviation
            if word in self.word_abbreviations:
                abbreviated_words.append(self.word_abbreviations[word])
            else:
                abbreviated_words.append(word)
        
        result = ' '.join(abbreviated_words)
        
        # If still too long, use ML to predict abbreviation ratio for remaining words
        if len(result) > target_length:
            # Start with the longest words
            words = result.split()
            word_indices = sorted(range(len(words)), key=lambda i: len(words[i]), reverse=True)
            
            for idx in word_indices:
                if len(result) <= target_length:
                    break
                    
                word = words[idx]
                # Skip short words and already abbreviated words
                if len(word) <= 3 or word in self.word_abbreviations.values():
                    continue
                
                # Predict abbreviation ratio for this word
                ratio = self.pipeline.predict([word])[0]
                
                # Apply intelligent abbreviation
                target_word_len = max(2, int(len(word) * ratio))
                if target_word_len < len(word):
                    # More advanced abbreviation strategy
                    if target_word_len >= 3:
                        # Keep prefix
                        words[idx] = word[:target_word_len]
                    elif len(word) > 4:
                        # For longer words, keep first and last letter
                        words[idx] = word[0] + word[-1]
                    else:
                        # Short words - keep as is
                        pass
                        
            # Recombine words
            result = ' '.join(words)
        
        # If still too long, truncate as last resort
        if len(result) > target_length:
            # Try to truncate at a word boundary
            last_space = result[:target_length-3].rfind(' ')
            if last_space > 0:
                result = result[:last_space] + "..."
            else:
                result = result[:target_length-3] + "..."
        
        return result
    
    def save(self, model_path):
        """Save the trained model to disk"""
        if not self.is_trained:
            print("Model not trained yet, nothing to save")
            return False
            
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save the model components
        joblib.dump({
            'pipeline': self.pipeline,
            'word_abbreviations': self.word_abbreviations,
            'phrase_abbreviations': self.phrase_abbreviations
        }, model_path)
        
        print(f"Model saved to {model_path}")
        return True
    
    def load(self, model_path):
        """Load a trained model from disk"""
        try:
            model_data = joblib.load(model_path)
            
            self.pipeline = model_data['pipeline']
            self.word_abbreviations = model_data['word_abbreviations']
            self.phrase_abbreviations = model_data['phrase_abbreviations']
            self.is_trained = True
            
            print(f"Model loaded from {model_path}")
            print(f"Loaded {len(self.word_abbreviations)} word abbreviations")
            print(f"Loaded {len(self.phrase_abbreviations)} phrase abbreviations")
            return True
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False

def train_and_save_model(training_data_path, model_output_path):
    """Train and save a hybrid abbreviation model"""
    # Load training data
    df = pd.read_csv(training_data_path)
    training_data = list(zip(df['original'], df['abbreviated']))
    
    # Create and train the model
    model = HybridAbbreviationModel()
    model.fit(training_data)
    
    # Save the model
    model.save(model_output_path)
    
    return model