"""
RAG System Utilities for Source Attribution

This module provides core functionality for:
- Loading datasets from CSV/JSON files
- LLM integration and response generation
- Utility function computation for document subsets
"""

import json
import ast
import os
import pandas as pd
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, skip


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
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-1B",
        device: Optional[str] = None,
        token: Optional[str] = None,
        max_input_tokens: Optional[int] = None,
        max_doc_tokens: Optional[int] = None
    ):
        """
        Initialize RAG system with LLM model.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to use ('cuda', 'cpu', or None for auto)
            token: HuggingFace token for authentication (or set HF_TOKEN env var)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.max_input_tokens = max_input_tokens
        self.max_doc_tokens = max_doc_tokens
        
        # Get token from parameter or environment variable
        hf_token = token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        
        print(f"Loading tokenizer for {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=hf_token
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # If not provided, default to model's advertised max length
        if self.max_input_tokens is None:
            self.max_input_tokens = getattr(self.tokenizer, "model_max_length", 2048)
        
        print(f"Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            low_cpu_mem_usage=True,
            token=hf_token
        )
        
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        
        self.model.eval()
        print(f"Model loaded successfully on {self.device}")
    
    def _tokenize_with_truncation(self, text: str):
        """
        Tokenize with an explicit max length and truncation to avoid over-length inputs.
        """
        return self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_input_tokens,
            truncation=True
        ).to(self.device)
    
    def truncate_documents(self, documents: List[str]) -> List[str]:
        """
        Truncate individual documents to max_doc_tokens if configured.
        """
        if not self.max_doc_tokens:
            return documents
        
        truncated = []
        for doc in documents:
            token_ids = self.tokenizer.encode(
                doc,
                add_special_tokens=False,
                max_length=self.max_doc_tokens,
                truncation=True
            )
            truncated.append(self.tokenizer.decode(token_ids, skip_special_tokens=True))
        return truncated
    
    def format_prompt(self, question: str, documents: List[str]) -> str:
        """
        Format prompt with context and question.
        
        Args:
            question: The question to answer
            documents: List of context documents
            
        Returns:
            Formatted prompt string
        """
        docs = self.truncate_documents(documents)
        context = "\n\n".join(docs)
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
        
        inputs = self._tokenize_with_truncation(prompt)
        
        # Create attention mask explicitly to avoid warnings
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is None:
            attention_mask = torch.ones_like(inputs["input_ids"])
        
        # Prepare generation kwargs
        generation_kwargs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "do_sample": do_sample,
        }
        
        # Only add temperature/top_p if sampling is enabled
        if do_sample:
            generation_kwargs["temperature"] = temperature
        
        with torch.no_grad():
            outputs = self.model.generate(**generation_kwargs)
        
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
        # Tokenize prompt to locate where target tokens start
        prompt_inputs = self._tokenize_with_truncation(prompt)
        prompt_length = prompt_inputs["input_ids"].shape[1]

        # Tokenize full text (prompt + target) with attention mask
        full_text = prompt + " " + target_text
        full_inputs = self._tokenize_with_truncation(full_text)
        input_ids = full_inputs["input_ids"]
        attention_mask = full_inputs.get("attention_mask", torch.ones_like(input_ids))

        seq_len = input_ids.shape[1]
        # Clip prompt length in case truncation cut into the prompt tokens
        prompt_length = min(prompt_length, seq_len)
        if seq_len <= prompt_length:
            return 0.0

        # Forward pass with attention mask
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # [1, seq_len, vocab]

        # Shift for causal LM: logits predict the next token
        logits = logits[:, :-1, :]            # positions 0..seq_len-2
        labels = input_ids[:, 1:]             # tokens 1..seq_len-1
        attn_shifted = attention_mask[:, 1:]  # align with labels

        # Mask to target tokens only (positions at/after first target token)
        target_mask = torch.zeros_like(labels, dtype=torch.bool)
        target_mask[:, prompt_length:] = True
        effective_mask = target_mask & (attn_shifted > 0)

        if effective_mask.sum() == 0:
            return 0.0

        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        # Gather log probs of the actual labels
        selected_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

        # Sum only over target tokens
        total_log_prob = selected_log_probs[effective_mask].sum().item()

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

