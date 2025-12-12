"""
Script to load and process synthetic data from the LLMX repository.

This script provides utilities to load CSV files containing questions, contexts, and answers
from the synthetic_data folder and extract the relevant information.
"""

import pandas as pd
import json
import ast
from pathlib import Path
from typing import List, Dict, Tuple


class SyntheticDataLoader:
    """Loads and processes synthetic RAG data from CSV files."""
    
    def __init__(self, data_dir: str = "data/"):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Path to the directory containing CSV files
        """
        self.data_dir = Path(data_dir) # gives path to directory where data is stored
        self.available_files = self._get_available_files() # a method that gets list of available CSV files
        
    def _get_available_files(self):
        """Get list of available CSV files."""
        if not self.data_dir.exists():
            return []
        return [f.name for f in self.data_dir.glob("*.csv")]
    
    def load_csv(self, filename):
        """
        Load a single CSV file and return a pandas DataFrame.
        
        Args:
            filename: Name of the CSV file to load
            
        Returns:
            pandas DataFrame containing the data
        """
        filepath = self.data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # Try to read with error handling for different CSV
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            # Try with semicolon delimiter
            try:
                df = pd.read_csv(filepath, sep=';')
            except:
                raise e
        
        # Reset index to ensure it's integer-based
        df = df.reset_index(drop=True)
        
        # Parse the context column - it stores lists as strings
        if 'context' in df.columns:
            # create new column with parsed context - apply the _parse_context method to each context row
            df['context_parsed'] = df['context'].apply(self._parse_context)         
        return df
    
    def _parse_context(self, context_str):
        """
        Convert context from string to list. 
        
        Args:
            context_str: Context as string (Python list literal)
            
        Returns:
            List of context strings or original value if parsing fails
        """
        # throw error if missing values
        if pd.isna(context_str):
            raise ValueError("Missing (NaN) value encountered in context field.")
        
        # If already a list, return it
        if isinstance(context_str, list):
            return context_str
        
        if isinstance(context_str, str):
            # Try ast.literal_eval first (for Python list literals with mixed quotes)
            try:
                parsed = ast.literal_eval(context_str)
                if isinstance(parsed, list):
                    return parsed
            except (ValueError, SyntaxError):
                raise ValueError(f"Failed to parse context string as a Python list: {context_str}")
        
        raise ValueError(f"Context field must be a list or string, got: {type(context_str)}")
    
    def load_all(self):
        """
        Load all available CSV files and return a dictionary of DataFrames.
        """
        data = {}
        for filename in self.available_files:
            try:
                data[filename] = self.load_csv(filename)
                print(f"Loaded {filename}: {len(data[filename])} rows")
            except Exception as e:
                print(f"Error loading {filename}: {e}")
        
        return data
    
    def get_question_context_pairs(self, df: pd.DataFrame) -> List[Tuple[str, List[str], str]]:
        """
        Extract question, context, and answer tuples from a DataFrame.
        
        Args:
            df: DataFrame containing question, context, and answer columns
            
        Returns:
            List of tuples (question, context_list, answer)
        """
        results = []
        
        for _, row in df.iterrows():
            question = row.get('question', '')
            answer = row.get('answer', '')
            
            # Get parsed context if available, otherwise parse it
            if 'context_parsed' in row:
                context = row['context_parsed']
            else:
                context = self._parse_context(row.get('context', []))
            
            results.append((question, context, answer))
        
        return results # returns list of tuples (question, context, answer) where context is a list of strings