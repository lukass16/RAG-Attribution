"""
Attribution Methods for RAG Source Attribution

This module implements various attribution methods for identifying influential documents:
- Exact Shapley values
- Shapley approximations (Monte Carlo, Permutation-based, Kernel SHAP)
- Baseline methods (Leave-One-Out, Gradient-based, Attention-based, Retrieval Score)
"""

import numpy as np
import torch
from typing import List, Dict, Callable, Optional
from itertools import combinations, permutations
import random
from shap import KernelExplainer
import warnings
warnings.filterwarnings('ignore')


def exact_shapley(
    documents: List[str],
    utility_function: Callable[[List[str]], float]
) -> Dict[str, float]:
    """
    Compute exact Shapley values for each document.
    
    Formula: φ_i = Σ_{S⊆D\{d_i}} [|S|!(|D|-|S|-1)!/|D|!] * [v(S∪{d_i}) - v(S)]
    
    Args:
        documents: List of all documents D
        utility_function: Function v(S) that takes a document subset and returns utility
        
    Returns:
        Dictionary mapping document index to Shapley value
    """
    n = len(documents)
    if n > 10:
        raise ValueError(f"Exact Shapley requires 2^{n} evaluations. For {n} documents, this is {2**n} calls. Use approximation methods instead.")
    
    shapley_values = {i: 0.0 for i in range(n)}
    
    # Iterate over all possible subsets
    for i in range(n):
        # All subsets that don't include document i
        other_docs = [j for j in range(n) if j != i]
        
        for subset_size in range(n):
            # All subsets of size subset_size from other_docs
            for subset_indices in combinations(other_docs, subset_size):
                subset = [documents[j] for j in subset_indices]
                subset_with_i = [documents[j] for j in subset_indices] + [documents[i]]
                
                # Compute marginal contribution
                v_with = utility_function(subset_with_i)
                v_without = utility_function(subset)
                marginal = v_with - v_without
                
                # Weight: |S|!(|D|-|S|-1)!/|D|!
                weight = (np.math.factorial(subset_size) * 
                         np.math.factorial(n - subset_size - 1)) / np.math.factorial(n)
                
                shapley_values[i] += weight * marginal
    
    return shapley_values


def monte_carlo_shapley(
    documents: List[str],
    utility_function: Callable[[List[str]], float],
    num_samples: int = 100
) -> Dict[str, float]:
    """
    Compute Shapley values using Monte Carlo sampling.
    
    Samples random coalitions and approximates marginal contributions.
    
    Args:
        documents: List of all documents D
        utility_function: Function v(S) that takes a document subset and returns utility
        num_samples: Number of random samples to use
        
    Returns:
        Dictionary mapping document index to approximate Shapley value
    """
    n = len(documents)
    shapley_values = {i: 0.0 for i in range(n)}
    
    for _ in range(num_samples):
        # Random permutation of documents
        perm = random.sample(range(n), n)
        
        # Compute marginal contributions along this permutation
        for i, doc_idx in enumerate(perm):
            # Documents before doc_idx in this permutation
            before = [documents[perm[j]] for j in range(i)]
            
            # Marginal contribution of doc_idx
            v_with = utility_function(before + [documents[doc_idx]])
            v_without = utility_function(before)
            marginal = v_with - v_without
            
            shapley_values[doc_idx] += marginal
    
    # Average over samples
    for i in range(n):
        shapley_values[i] /= num_samples
    
    return shapley_values


def permutation_shapley(
    documents: List[str],
    utility_function: Callable[[List[str]], float],
    num_permutations: int = 50
) -> Dict[str, float]:
    """
    Compute Shapley values using permutation-based sampling.
    
    Similar to Monte Carlo but explicitly samples permutations.
    
    Args:
        documents: List of all documents D
        utility_function: Function v(S) that takes a document subset and returns utility
        num_permutations: Number of random permutations to sample
        
    Returns:
        Dictionary mapping document index to approximate Shapley value
    """
    n = len(documents)
    shapley_values = {i: 0.0 for i in range(n)}
    
    for _ in range(num_permutations):
        # Random permutation
        perm = list(range(n))
        random.shuffle(perm)
        
        # Compute marginal contributions
        for i, doc_idx in enumerate(perm):
            before = [documents[perm[j]] for j in range(i)]
            
            v_with = utility_function(before + [documents[doc_idx]])
            v_without = utility_function(before)
            marginal = v_with - v_without
            
            shapley_values[doc_idx] += marginal
    
    # Average
    for i in range(n):
        shapley_values[i] /= num_permutations
    
    return shapley_values


def kernel_shap(
    documents: List[str],
    utility_function: Callable[[List[str]], float],
    num_samples: int = 100
) -> Dict[str, float]:
    """
    Compute Shapley values using Kernel SHAP (weighted linear regression).
    
    Uses SHAP library's KernelExplainer for efficient approximation.
    
    Args:
        documents: List of all documents D
        utility_function: Function v(S) that takes a document subset and returns utility
        num_samples: Number of samples for Kernel SHAP
        
    Returns:
        Dictionary mapping document index to approximate Shapley value
    """
    n = len(documents)
    
    # Wrapper function for SHAP (takes binary vector indicating which documents are included)
    def shap_wrapper(binary_vectors):
        # Handle both single vector and batch
        if len(binary_vectors.shape) == 1:
            binary_vectors = binary_vectors.reshape(1, -1)
        
        results = []
        for binary_vector in binary_vectors:
            subset = [documents[i] for i in range(n) if binary_vector[i] == 1]
            results.append(utility_function(subset))
        
        return np.array(results)
    
    # Create explainer with all zeros baseline (empty set)
    explainer = KernelExplainer(shap_wrapper, np.zeros((1, n)))
    
    # Compute SHAP values for all documents included
    shap_values = explainer.shap_values(np.ones((1, n)), nsamples=num_samples)
    
    # Convert to dictionary (handle both list and array outputs)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    if len(shap_values.shape) > 1:
        shap_values = shap_values[0]
    
    result = {i: float(shap_values[i]) for i in range(n)}
    
    return result


def leave_one_out(
    documents: List[str],
    utility_function: Callable[[List[str]], float]
) -> Dict[str, float]:
    """
    Compute attribution using Leave-One-Out method.
    
    Attribution = v(D) - v(D\{d_i})
    
    Args:
        documents: List of all documents D
        utility_function: Function v(S) that takes a document subset and returns utility
        
    Returns:
        Dictionary mapping document index to attribution score
    """
    n = len(documents)
    v_all = utility_function(documents)
    
    attributions = {}
    for i in range(n):
        subset = [documents[j] for j in range(n) if j != i]
        v_without = utility_function(subset)
        attributions[i] = v_all - v_without
    
    return attributions


def gradient_attribution(
    documents: List[str],
    question: str,
    target_response: str,
    model,
    tokenizer,
    device: str
) -> Dict[str, float]:
    """
    Compute gradient-based attribution.
    
    Computes gradient of utility w.r.t. document embeddings.
    
    Args:
        documents: List of all documents D
        question: The question Q
        target_response: Target response Rtarget
        model: LLM model
        tokenizer: Tokenizer
        device: Device to use
        
    Returns:
        Dictionary mapping document index to gradient-based attribution score
    """
    from rag_system import RAGSystem
    
    # This is a placeholder - full implementation would require
    # computing gradients w.r.t. document embeddings
    # For now, return zeros as placeholder
    n = len(documents)
    attributions = {i: 0.0 for i in range(n)}
    
    # TODO: Implement full gradient computation
    # This would involve:
    # 1. Creating embeddings for each document
    # 2. Computing utility with gradients enabled
    # 3. Backpropagating to get gradients
    # 4. Using gradient magnitude as attribution
    
    return attributions


def attention_attribution(
    documents: List[str],
    question: str,
    model,
    tokenizer,
    device: str
) -> Dict[str, float]:
    """
    Compute attention-based attribution.
    
    Extracts attention weights and aggregates to document level.
    
    Args:
        documents: List of all documents D
        question: The question Q
        model: LLM model
        tokenizer: Tokenizer
        device: Device to use
        
    Returns:
        Dictionary mapping document index to attention-based attribution score
    """
    from rag_system import RAGSystem
    
    rag = RAGSystem.__new__(RAGSystem)
    rag.model = model
    rag.tokenizer = tokenizer
    rag.device = device
    
    prompt = rag.format_prompt(question, documents)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Tokenize prompt to get token positions
    prompt_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
    # Find document token ranges by searching for document content in tokens
    # Simple approach: split prompt and find document boundaries
    doc_token_ranges = []
    prompt_parts = prompt.split("\n\n")
    # Format is: "Context: {doc1}\n\n{doc2}\n\n...\n\nQuestion: {question}\n\nAnswer:"
    
    # Find where "Context:" starts
    context_start_idx = None
    for i, part in enumerate(prompt_parts):
        if part.startswith("Context:"):
            context_start_idx = i
            break
    
    if context_start_idx is not None:
        # Documents are between "Context:" and "Question:"
        doc_parts = prompt_parts[context_start_idx + 1:-2]  # Exclude "Context:" and "Question:" and "Answer:"
        
        # Tokenize to find positions (simplified - approximate)
        current_pos = 0
        for doc in documents:
            # Find approximate position by tokenizing document
            doc_tokens = tokenizer(doc, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
            doc_len = len(doc_tokens)
            doc_token_ranges.append((current_pos, current_pos + doc_len))
            current_pos += doc_len + 2  # Approximate separator tokens
    else:
        # Fallback: equal distribution
        seq_len = inputs["input_ids"].shape[1]
        doc_len = seq_len // len(documents)
        for i in range(len(documents)):
            doc_token_ranges.append((i * doc_len, (i + 1) * doc_len))
    
    # Get attention weights
    with torch.no_grad():
        outputs = model(inputs["input_ids"], output_attentions=True)
    
    attentions = outputs.attentions
    n_layers = len(attentions)
    seq_len = inputs["input_ids"].shape[1]
    
    # Average attention across layers and heads
    # Focus on attention from last token (generation position) to document tokens
    n_docs = len(documents)
    doc_attentions = np.zeros(n_docs)
    
    for layer_attn in attentions:
        # Shape: (batch, heads, seq_len, seq_len)
        layer_attn = layer_attn[0].mean(dim=0)  # Average over heads: (seq_len, seq_len)
        
        # Get attention from last token to all tokens
        if layer_attn.shape[0] > 0 and seq_len > 0:
            last_token_idx = min(seq_len - 1, layer_attn.shape[0] - 1)
            last_token_attn = layer_attn[last_token_idx, :].cpu().numpy()
            
            # Aggregate attention to each document
            for i, (start, end) in enumerate(doc_token_ranges):
                start = min(int(start), len(last_token_attn) - 1)
                end = min(int(end), len(last_token_attn))
                if start < end:
                    doc_attentions[i] += last_token_attn[start:end].sum()
    
    # Average over layers
    if n_layers > 0:
        doc_attentions /= n_layers
    
    # Convert to dictionary
    attributions = {i: float(doc_attentions[i]) for i in range(n_docs)}
    
    return attributions


def retrieval_score_attribution(
    documents: List[str],
    question: str,
    retrieval_scores: Optional[List[float]] = None
) -> Dict[str, float]:
    """
    Compute attribution using retrieval similarity scores.
    
    Simple baseline that uses original retrieval scores.
    
    Args:
        documents: List of all documents D
        question: The question Q (for potential re-computation)
        retrieval_scores: Optional pre-computed retrieval scores
        
    Returns:
        Dictionary mapping document index to retrieval score
    """
    n = len(documents)
    
    if retrieval_scores is not None:
        if len(retrieval_scores) != n:
            raise ValueError(f"Number of retrieval scores ({len(retrieval_scores)}) doesn't match number of documents ({n})")
        attributions = {i: float(retrieval_scores[i]) for i in range(n)}
    else:
        # If no scores provided, return uniform scores (placeholder)
        attributions = {i: 1.0 / n for i in range(n)}
    
    return attributions

