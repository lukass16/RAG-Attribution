"""
RAG System Utilities for Source Attribution

This module provides core functionality for:
- Loading datasets from CSV/JSON files
- LLM integration and response generation
- Utility function computation for document subsets
"""

import json
import ast
import pandas as pd
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')


def load_dataset(file_path: str) -> List[Dict]:
    """
    Load dataset from CSV or JSON file.
    
    Args:
        file_path: Path to CSV or JSON file
        
    Returns:
        List of dictionaries with keys: question, documents, answer, document_ids
    """
    if file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        result = []
        for item in data:
            result.append({
                'question': item['question'],
                'documents': item['documents'],
                'answer': item.get('answer', ''),
                'document_ids': [chr(65 + i) for i in range(len(item['documents']))]  # A, B, C, ...
            })
        return result
    
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        result = []
        
        for _, row in df.iterrows():
            # Parse context column (can be string representation of list)
            context = row.get('context', [])
            if isinstance(context, str):
                try:
                    # Try to parse as Python literal
                    context = ast.literal_eval(context)
                except:
                    # If that fails, try JSON
                    try:
                        context = json.loads(context)
                    except:
                        # Last resort: split by comma (simple case)
                        context = [c.strip() for c in context.split(',')]
            
            if not isinstance(context, list):
                context = [context]
            
            # Get document IDs if available, otherwise generate them
            doc_ids = row.get('id', None)
            if doc_ids is not None:
                if isinstance(doc_ids, str):
                    try:
                        doc_ids = ast.literal_eval(doc_ids)
                    except:
                        doc_ids = [chr(65 + i) for i in range(len(context))]
                elif not isinstance(doc_ids, list):
                    doc_ids = [chr(65 + i) for i in range(len(context))]
            else:
                doc_ids = [chr(65 + i) for i in range(len(context))]
            
            result.append({
                'question': row.get('question', ''),
                'documents': context,
                'answer': row.get('answer', ''),
                'document_ids': doc_ids
            })
        
        return result
    
    else:
        raise ValueError(f"Unsupported file format: {file_path}")


class RAGSystem:
    """
    RAG System for generating responses and computing utilities.
    """
    
    def __init__(self, model_name: str = "meta-llama/Llama-3.2-1B", device: Optional[str] = None):
        """
        Initialize RAG system with LLM model.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        
        print(f"Loading tokenizer for {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            low_cpu_mem_usage=True
        )
        
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        
        self.model.eval()
        print(f"Model loaded successfully on {self.device}")
    
    def format_prompt(self, question: str, documents: List[str]) -> str:
        """
        Format prompt with context and question.
        
        Args:
            question: The question to answer
            documents: List of context documents
            
        Returns:
            Formatted prompt string
        """
        context = "\n\n".join(documents)
        prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        return prompt
    
    def generate_response(
        self, 
        question: str, 
        documents: List[str], 
        max_new_tokens: int = 50,
        temperature: float = 0.7,
        do_sample: bool = True
    ) -> str:
        """
        Generate response given question and document subset.
        
        Args:
            question: The question to answer
            documents: List of context documents (subset S)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            
        Returns:
            Generated response text
        """
        prompt = self.format_prompt(question, documents)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode only the generated part (excluding prompt)
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return response.strip()
    
    def generate_target_response(
        self, 
        question: str, 
        all_documents: List[str],
        max_new_tokens: int = 50
    ) -> str:
        """
        Generate target response Rtarget using ALL documents.
        This is the "gold standard" response used for utility computation.
        
        Args:
            question: The question to answer
            all_documents: Complete list of all documents D
            
        Returns:
            Target response Rtarget
        """
        return self.generate_response(
            question, 
            all_documents, 
            max_new_tokens=max_new_tokens,
            do_sample=False  # Use greedy decoding for reproducibility
        )
    
    def compute_log_probability(
        self,
        prompt: str,
        target_text: str
    ) -> float:
        """
        Compute log probability of generating target_text given prompt.
        
        Args:
            prompt: Input prompt (question + context)
            target_text: Target text to compute probability for
            
        Returns:
            Sum of log probabilities for all tokens in target_text
        """
        # Tokenize prompt
        prompt_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_length = prompt_inputs["input_ids"].shape[1]
        
        # Tokenize full text (prompt + target)
        full_text = prompt + " " + target_text
        full_inputs = self.tokenizer(full_text, return_tensors="pt").to(self.device)
        
        # Get target token IDs
        target_ids = full_inputs["input_ids"][0][prompt_length:]
        
        if len(target_ids) == 0:
            return 0.0
        
        # Forward pass to get logits
        with torch.no_grad():
            outputs = self.model(full_inputs["input_ids"])
            logits = outputs.logits
        
        # Compute log probabilities for each target token
        log_probs = []
        for i, token_id in enumerate(target_ids):
            # Get logits for this position (the position where this token is predicted)
            pos_idx = prompt_length + i
            if pos_idx >= logits.shape[1]:
                break
            
            token_logits = logits[0, pos_idx, :]
            log_probs_token = torch.nn.functional.log_softmax(token_logits, dim=-1)
            log_prob = log_probs_token[token_id].item()
            log_probs.append(log_prob)
        
        # Sum log probabilities
        total_log_prob = sum(log_probs)
        
        return total_log_prob
    
    def compute_utility(
        self,
        question: str,
        document_subset: List[str],
        target_response: str
    ) -> float:
        """
        Compute utility function v(S) = log P(Rtarget | Q, S).
        
        Measures how well document subset S supports generating target_response.
        
        Args:
            question: The question Q
            document_subset: Document subset S
            target_response: Target response Rtarget
            
        Returns:
            Utility score (higher = better support for Rtarget)
        """
        # Format prompt with subset
        prompt = self.format_prompt(question, document_subset)
        
        # Compute log probability of generating target_response
        log_prob = self.compute_log_probability(prompt, target_response)
        
        return log_prob
    
    def compute_utility_batch(
        self,
        question: str,
        document_subsets: List[List[str]],
        target_response: str
    ) -> List[float]:
        """
        Compute utility for multiple document subsets (for efficiency).
        
        Args:
            question: The question Q
            document_subsets: List of document subsets
            target_response: Target response Rtarget
            
        Returns:
            List of utility scores
        """
        utilities = []
        for subset in document_subsets:
            util = self.compute_utility(question, subset, target_response)
            utilities.append(util)
        return utilities

