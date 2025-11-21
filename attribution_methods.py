"""
Attribution Methods for RAG Source Attribution

This module implements various attribution methods for identifying influential documents:
- Exact Shapley values
- Shapley approximations (Monte Carlo, Permutation-based, Kernel SHAP)
- Baseline methods (Leave-One-Out, Gradient-based, Attention-based, Retrieval Score)
"""

import numpy as np
import torch
from typing import List, Dict, Callable, Optional, Tuple
from itertools import combinations, permutations
import random
from shap import KernelExplainer
import warnings
warnings.filterwarnings('ignore')

try:
    from captum.attr import IntegratedGradients
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False
    warnings.warn("Captum not available. Integrated gradients will not work. Install with: pip install captum")


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


def _find_document_token_ranges(
    documents: List[str],
    question: str,
    tokenizer,
    prompt: str,
    prompt_tokens: List[str]
) -> List[Tuple[int, int]]:
    """
    Helper function to find token ranges for each document in the prompt.
    
    Args:
        documents: List of documents
        question: The question
        tokenizer: Tokenizer
        prompt: Full formatted prompt
        prompt_tokens: List of token strings
        
    Returns:
        List of (start_idx, end_idx) tuples for each document
    """
    doc_token_ranges = []
    
    # Find where "Context:" starts in tokens
    context_token_idx = None
    for i, token in enumerate(prompt_tokens):
        if "context" in token.lower() or (i > 0 and prompt_tokens[i-1] == ":"):
            context_token_idx = i + 1
            break
    
    if context_token_idx is None:
        context_token_idx = 0
    
    # Tokenize each document to find its position
    current_pos = context_token_idx
    for doc in documents:
        # Tokenize document
        doc_tokens = tokenizer(doc, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        doc_len = len(doc_tokens)
        
        # Find where this document appears in the prompt tokens
        # Simple approach: search for first token of document
        start_idx = None
        if len(doc_tokens) > 0:
            first_doc_token_id = doc_tokens[0].item()
            # Search from current_pos onwards
            for i in range(current_pos, len(prompt_tokens)):
                if i < len(tokenizer.convert_tokens_to_ids(prompt_tokens)):
                    token_id = tokenizer.convert_tokens_to_ids([prompt_tokens[i]])[0]
                    if token_id == first_doc_token_id:
                        start_idx = i
                        break
        
        if start_idx is None:
            start_idx = current_pos
        
        end_idx = min(start_idx + doc_len, len(prompt_tokens))
        doc_token_ranges.append((start_idx, end_idx))
        current_pos = end_idx + 2  # Skip separators
    
    return doc_token_ranges


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
    Compute gradient-based attribution for documents.
    
    Computes gradient of utility w.r.t. token embeddings, then aggregates to document level.
    
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
    
    rag = RAGSystem.__new__(RAGSystem)
    rag.model = model
    rag.tokenizer = tokenizer
    rag.device = device
    
    # Format prompt with all documents
    prompt = rag.format_prompt(question, documents)
    
    # Tokenize to find document token ranges
    prompt_inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_tokens = tokenizer.convert_ids_to_tokens(prompt_inputs["input_ids"][0])
    
    # Find which tokens belong to which document
    doc_token_ranges = _find_document_token_ranges(
        documents, question, tokenizer, prompt, prompt_tokens
    )
    
    # Tokenize full text (prompt + target) for utility computation
    full_text = prompt + " " + target_response
    full_inputs = tokenizer(full_text, return_tensors="pt").to(device)
    prompt_length = prompt_inputs["input_ids"].shape[1]
    target_ids = full_inputs["input_ids"][0][prompt_length:]
    
    if len(target_ids) == 0:
        return {i: 0.0 for i in range(len(documents))}
    
    # Get embeddings layer
    embeddings_layer = model.get_input_embeddings()
    embedding_matrix = embeddings_layer.weight
    
    # Create token embeddings with gradients enabled
    with torch.enable_grad():
        # Get input token IDs for the prompt
        input_ids = prompt_inputs["input_ids"]
        token_embeddings = embedding_matrix[input_ids].detach().clone()
        token_embeddings.requires_grad_(True)
        
        # Forward pass
        outputs = model(inputs_embeds=token_embeddings)
        logits = outputs.logits
        
        # Compute utility: sum of log probabilities for target tokens
        total_log_prob = 0.0
        for i, token_id in enumerate(target_ids):
            pos_idx = prompt_length + i
            if pos_idx >= logits.shape[1]:
                break
            
            token_logits = logits[0, pos_idx, :]
            log_probs_token = torch.nn.functional.log_softmax(token_logits, dim=-1)
            log_prob = log_probs_token[token_id]
            total_log_prob += log_prob
        
        # Backward pass
        total_log_prob.backward()
        
        # Get gradients
        gradients = token_embeddings.grad
        
        if gradients is None:
            return {i: 0.0 for i in range(len(documents))}
        
        # Aggregate gradients to document level
        doc_gradients = []
        for start_idx, end_idx in doc_token_ranges:
            if start_idx < gradients.shape[1] and end_idx <= gradients.shape[1]:
                doc_grad = gradients[0, start_idx:end_idx, :]
                # Use L2 norm of aggregated gradients
                doc_grad_norm = doc_grad.norm(dim=-1).sum().item()
                doc_gradients.append(doc_grad_norm)
            else:
                doc_gradients.append(0.0)
    
    # Convert to dictionary
    attributions = {i: float(doc_gradients[i]) for i in range(len(documents))}
    
    return attributions


def integrated_gradients_attribution(
    documents: List[str],
    question: str,
    target_response: str,
    model,
    tokenizer,
    device: str,
    baseline_type: str = 'zero',
    n_steps: int = 50
) -> Dict[str, float]:
    """
    Compute integrated gradients attribution for documents.
    
    Uses Captum's IntegratedGradients to compute path-integral based attribution,
    then aggregates to document level.
    
    Args:
        documents: List of all documents D
        question: The question Q
        target_response: Target response Rtarget
        model: LLM model
        tokenizer: Tokenizer
        device: Device to use
        baseline_type: Type of baseline ('zero' or 'pad')
        n_steps: Number of steps for integral approximation
        
    Returns:
        Dictionary mapping document index to integrated gradients attribution score
    """
    if not CAPTUM_AVAILABLE:
        raise ImportError("Captum is required for integrated gradients. Install with: pip install captum")
    
    from rag_system import RAGSystem
    
    rag = RAGSystem.__new__(RAGSystem)
    rag.model = model
    rag.tokenizer = tokenizer
    rag.device = device
    
    # Format prompt with all documents
    prompt = rag.format_prompt(question, documents)
    
    # Tokenize to find document token ranges
    prompt_inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_tokens = tokenizer.convert_ids_to_tokens(prompt_inputs["input_ids"][0])
    
    # Find which tokens belong to which document
    doc_token_ranges = _find_document_token_ranges(
        documents, question, tokenizer, prompt, prompt_tokens
    )
    
    # Tokenize full text (prompt + target) for utility computation
    full_text = prompt + " " + target_response
    full_inputs = tokenizer(full_text, return_tensors="pt").to(device)
    prompt_length = prompt_inputs["input_ids"].shape[1]
    target_ids = full_inputs["input_ids"][0][prompt_length:]
    
    if len(target_ids) == 0:
        return {i: 0.0 for i in range(len(documents))}
    
    # Get embeddings
    embeddings_layer = model.get_input_embeddings()
    embedding_matrix = embeddings_layer.weight
    input_ids = prompt_inputs["input_ids"]
    input_embeddings = embedding_matrix[input_ids].detach().clone()
    
    # Create baseline embeddings
    if baseline_type == 'zero':
        baseline_embeddings = torch.zeros_like(input_embeddings)
    else:  # 'pad'
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        baseline_ids = torch.full_like(input_ids, pad_token_id)
        baseline_embeddings = embedding_matrix[baseline_ids].detach().clone()
    
    # Define forward function for IntegratedGradients
    # This computes utility (log probability of target response)
    def model_forward(embeddings, attention_mask=None):
        """Forward function that returns utility (log prob of target)."""
        outputs = model(inputs_embeds=embeddings, attention_mask=attention_mask)
        logits = outputs.logits
        
        # Compute utility: sum of log probabilities for target tokens
        log_probs_list = []
        for i, token_id in enumerate(target_ids):
            pos_idx = prompt_length + i
            if pos_idx >= logits.shape[1]:
                break
            
            token_logits = logits[0, pos_idx, :]
            log_probs_token = torch.nn.functional.log_softmax(token_logits, dim=-1)
            log_prob = log_probs_token[token_id]
            log_probs_list.append(log_prob)
        
        if len(log_probs_list) == 0:
            return torch.tensor(0.0, device=device).unsqueeze(0)
        
        # Sum all log probabilities
        total_log_prob = sum(log_probs_list)
        
        # Return as a single value (IG expects scalar output)
        return total_log_prob.unsqueeze(0)
    
    # Initialize IntegratedGradients
    ig = IntegratedGradients(model_forward)
    
    # Compute attributions
    attributions = ig.attribute(
        inputs=input_embeddings,
        baselines=baseline_embeddings,
        target=None,  # We're computing utility directly
        n_steps=n_steps,
        internal_batch_size=1
    )
    
    # Aggregate attributions to document level
    doc_attributions = []
    for start_idx, end_idx in doc_token_ranges:
        if start_idx < attributions.shape[1] and end_idx <= attributions.shape[1]:
            doc_attr = attributions[0, start_idx:end_idx, :]
            # Use L2 norm of aggregated attributions
            doc_attr_norm = doc_attr.norm(dim=-1).sum().item()
            doc_attributions.append(doc_attr_norm)
        else:
            doc_attributions.append(0.0)
    
    # Convert to dictionary
    result = {i: float(doc_attributions[i]) for i in range(len(documents))}
    
    return result


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

