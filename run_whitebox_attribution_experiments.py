#!/usr/bin/env python3
"""
Whitebox Attribution Experiment Script for RAG Source Attribution Analysis

This script runs whitebox attribution experiments (gradient-based and attention-based)
across multiple queries and datasets, computes evaluation metrics, and saves results
in the same format as run_attribution_experiments.py for visualization compatibility.

Whitebox methods implemented:
- gradient: Gradient-based attribution (gradient of utility w.r.t. embeddings)
- integrated_gradients: Integrated Gradients attribution (path-integral method)
- attention: Attention-based attribution (aggregated attention weights)
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import argparse
from datetime import datetime
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from rag_system import RAGSystem, load_dataset
import attribution_methods as attr


def compute_top_k_accuracy(attributions: List[float], doc_ids: List[str], 
                           k: int = 2, correct_docs: List[str] = ['A', 'B']) -> bool:
    """
    Check if the top-k documents match the expected correct documents.
    
    Args:
        attributions: List of attribution scores
        doc_ids: List of document IDs
        k: Number of top documents to check
        correct_docs: Expected correct document IDs (default: ['A', 'B'])
    
    Returns:
        True if top-k contains all correct documents
    """
    # Get sorted indices by ABSOLUTE attribution score (descending)
    # Use absolute value because large impact (positive or negative) = important
    sorted_indices = sorted(range(len(doc_ids)), 
                           key=lambda i: abs(attributions[i]), 
                           reverse=True)
    
    # Get top-k document IDs
    top_k_docs = [doc_ids[i] for i in sorted_indices[:k]]
    
    # Check if all correct docs are in top-k
    return set(correct_docs).issubset(set(top_k_docs))


def compute_rank_of_correct_docs(attributions: List[float], doc_ids: List[str],
                                 correct_docs: List[str] = ['A', 'B']) -> Dict[str, int]:
    """
    Compute the rank of each correct document.
    
    Returns:
        Dictionary mapping document ID to its rank (1-indexed)
    """
    # Sort by ABSOLUTE value - documents with largest impact rank highest
    sorted_indices = sorted(range(len(doc_ids)),
                           key=lambda i: abs(attributions[i]),
                           reverse=True)
    
    ranks = {}
    for rank, idx in enumerate(sorted_indices, 1):
        doc_id = doc_ids[idx]
        if doc_id in correct_docs:
            ranks[doc_id] = rank
    
    return ranks


def run_whitebox_attribution_experiment(
    dataset_path: str,
    model_name: str = "meta-llama/Llama-3.2-1B",
    device: Optional[str] = None,
    max_queries: Optional[int] = None,
    methods: Optional[List[str]] = None,
    ig_baseline_type: str = 'zero',
    ig_n_steps: int = 50,
    attn_implementation: Optional[str] = None
) -> Dict:
    """
    Run whitebox attribution experiment on a dataset.
    
    Args:
        dataset_path: Path to dataset CSV/JSON file
        model_name: HuggingFace model name
        device: Device to use ('cuda', 'cpu', or None for auto)
        max_queries: Maximum number of queries to process (None = all)
        methods: List of whitebox attribution methods to use (None = all)
        ig_baseline_type: Baseline type for integrated gradients ('zero' or 'pad')
        ig_n_steps: Number of steps for integrated gradients approximation
        attn_implementation: Attention implementation ('eager' required for attention method)
    
    Returns:
        Dictionary containing results and metrics
    """
    print(f"\n{'='*80}")
    print(f"Running WHITEBOX attribution experiment on: {dataset_path}")
    print(f"{'='*80}\n")
    
    run_started = datetime.now()
    run_started_ts = run_started.isoformat()
    
    # Auto-select eager attention if 'attention' method will be used
    effective_attn_impl = attn_implementation
    # Check if attention method will be used (either in methods list or in defaults)
    will_use_attention = (methods is None) or ('attention' in methods)
    if will_use_attention and effective_attn_impl is None:
        print("Note: 'attention' method requires eager attention. Auto-setting attn_implementation='eager'")
        effective_attn_impl = 'eager'
    
    # Initialize RAG system
    print("Initializing RAG system...")
    rag = RAGSystem(model_name=model_name, device=device, attn_implementation=effective_attn_impl)
    print("RAG system ready!\n")
    
    # Load dataset
    print(f"Loading dataset from {dataset_path}...")
    dataset = load_dataset(dataset_path)
    if max_queries:
        dataset = dataset[:max_queries]
    print(f"Loaded {len(dataset)} query-document pairs\n")
    
    # Generate target responses for all queries
    print("Generating target responses...")
    target_responses = []
    processed_documents = []
    for i, item in enumerate(dataset):
        print(f"  Processing query {i+1}/{len(dataset)}...", end='\r')
        docs_proc = rag.truncate_documents(item['documents'])
        processed_documents.append(docs_proc)
        # Prefer gold answer when provided; fall back to model-generated target
        if item.get('answer'):
            rtarget = str(item['answer'])
        else:
            rtarget = rag.generate_target_response(
                question=item['question'],
                all_documents=docs_proc,
                max_new_tokens=50
            )
        target_responses.append(rtarget)
    print(f"\nGenerated {len(target_responses)} target responses\n")
    
    # Default whitebox methods if not specified
    if methods is None:
        methods = ['gradient', 'attention']
        # Only include integrated_gradients if captum is available
        if attr.CAPTUM_AVAILABLE:
            methods.append('integrated_gradients')
        else:
            print("Note: Captum not available, skipping integrated_gradients method")
    
    # Results storage
    all_results = []
    
    # Process each query
    for query_idx, item in enumerate(dataset):
        print(f"\nProcessing query {query_idx + 1}/{len(dataset)}")
        print(f"Question: {item['question'][:80]}...")
        
        target_response = target_responses[query_idx]
        docs_proc = processed_documents[query_idx]
        doc_ids = item['document_ids']
        n_docs = len(docs_proc)
        gold_docs = item.get('gold_docs', ['A', 'B'])
        
        query_results = {
            'query_idx': query_idx,
            'question': item['question'],
            'expected_answer': item.get('answer', ''),
            'target_response': target_response,
            'n_documents': n_docs,
            'document_ids': doc_ids,
            'documents_raw': item['documents'],
            'documents': docs_proc,
            'document_lengths_raw': [len(doc) for doc in item['documents']],
            'document_lengths': [len(doc) for doc in docs_proc],
            'attributions': {},
            'attributions_abs': {},
            'timings_seconds': {}
        }
        
        # Compute attributions for each whitebox method
        for method_name in methods:
            try:
                method_start = time.perf_counter()
                
                if method_name == 'gradient':
                    attributions = attr.gradient_attribution(
                        documents=docs_proc,
                        question=item['question'],
                        target_response=target_response,
                        model=rag.model,
                        tokenizer=rag.tokenizer,
                        device=rag.device
                    )
                
                elif method_name == 'integrated_gradients':
                    if not attr.CAPTUM_AVAILABLE:
                        print(f"    Skipping integrated_gradients - Captum not available")
                        continue
                    attributions = attr.integrated_gradients_attribution(
                        documents=docs_proc,
                        question=item['question'],
                        target_response=target_response,
                        model=rag.model,
                        tokenizer=rag.tokenizer,
                        device=rag.device,
                        baseline_type=ig_baseline_type,
                        n_steps=ig_n_steps
                    )
                
                elif method_name == 'attention':
                    attributions = attr.attention_attribution(
                        documents=docs_proc,
                        question=item['question'],
                        model=rag.model,
                        tokenizer=rag.tokenizer,
                        device=rag.device
                    )
                
                else:
                    print(f"    Unknown whitebox method: {method_name}")
                    continue
                
                # Convert attributions to list format (handle dict, array, or list)
                if isinstance(attributions, dict):
                    # Convert dict {index: value} to list [value_0, value_1, ...]
                    attributions_list = [attributions.get(i, 0.0) for i in range(len(docs_proc))]
                elif isinstance(attributions, np.ndarray):
                    attributions_list = attributions.tolist()
                else:
                    attributions_list = list(attributions)
                
                # Store attributions
                query_results['attributions'][method_name] = attributions_list
                query_results['attributions_abs'][method_name] = [abs(v) for v in attributions_list]
                
                # Use list for metric computation
                attributions_for_metrics = attributions_list
                
                # Compute metrics
                top_2_accuracy = compute_top_k_accuracy(attributions_for_metrics, doc_ids, k=2, correct_docs=gold_docs)
                ranks = compute_rank_of_correct_docs(attributions_for_metrics, doc_ids, correct_docs=gold_docs)
                
                query_results[f'{method_name}_top2_accuracy'] = top_2_accuracy
                query_results[f'{method_name}_rank_A'] = ranks.get('A', None)
                query_results[f'{method_name}_rank_B'] = ranks.get('B', None)
                
                elapsed = time.perf_counter() - method_start
                print(f"    {method_name}: Top-2 accuracy = {top_2_accuracy}, Rank A = {ranks.get('A')}, Rank B = {ranks.get('B')} ({elapsed:.2f}s)")
                query_results['timings_seconds'][method_name] = elapsed
                
            except Exception as e:
                print(f"    Error computing {method_name}: {e}")
                import traceback
                traceback.print_exc()
                query_results['attributions'][method_name] = None
                query_results['timings_seconds'][method_name] = None
        
        all_results.append(query_results)
    
    # Compute aggregate metrics
    aggregate_metrics = {}
    for method_name in methods:
        top2_accuracies = [
            r.get(f'{method_name}_top2_accuracy', False) 
            for r in all_results 
            if f'{method_name}_top2_accuracy' in r
        ]
        ranks_A = [
            r.get(f'{method_name}_rank_A') 
            for r in all_results 
            if f'{method_name}_rank_A' in r and r.get(f'{method_name}_rank_A') is not None
        ]
        ranks_B = [
            r.get(f'{method_name}_rank_B') 
            for r in all_results 
            if f'{method_name}_rank_B' in r and r.get(f'{method_name}_rank_B') is not None
        ]
        
        aggregate_metrics[method_name] = {
            'top2_accuracy': np.mean(top2_accuracies) if top2_accuracies else None,
            'mean_rank_A': np.mean(ranks_A) if ranks_A else None,
            'mean_rank_B': np.mean(ranks_B) if ranks_B else None,
            'n_queries': len(top2_accuracies)
        }
    
    runtime_seconds = (datetime.now() - run_started).total_seconds()
    
    return {
        'dataset_path': dataset_path,
        'model_name': model_name,
        'experiment_type': 'whitebox',
        'timestamp': datetime.now().isoformat(),
        'run_started': run_started_ts,
        'runtime_seconds': runtime_seconds,
        'n_queries': len(dataset),
        'methods': methods,
        'run_config': {
            'model_name': model_name,
            'device': rag.device,
            'max_queries': max_queries,
            'dataset_path': dataset_path,
            'tokenizer_max_length': getattr(rag, "max_input_tokens", None),
            'max_doc_tokens': getattr(rag, "max_doc_tokens", None),
            'ig_baseline_type': ig_baseline_type,
            'ig_n_steps': ig_n_steps,
            'attn_implementation': attn_implementation,
        },
        'results': all_results,
        'aggregate_metrics': aggregate_metrics
    }


def save_results(results: Dict, output_dir: str = "results"):
    """Save experiment results to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = Path(results['dataset_path']).stem
    
    # Save full results as JSON (with whitebox prefix for clarity)
    json_path = os.path.join(output_dir, f"whitebox_{dataset_name}_{timestamp}_full.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved full results to: {json_path}")
    
    # Save summary metrics as CSV
    metrics_data = []
    for method_name, metrics in results['aggregate_metrics'].items():
        metrics_data.append({
            'method': method_name,
            'top2_accuracy': metrics['top2_accuracy'],
            'mean_rank_A': metrics['mean_rank_A'],
            'mean_rank_B': metrics['mean_rank_B'],
            'n_queries': metrics['n_queries']
        })
    
    metrics_df = pd.DataFrame(metrics_data)
    csv_path = os.path.join(output_dir, f"whitebox_{dataset_name}_{timestamp}_metrics.csv")
    metrics_df.to_csv(csv_path, index=False)
    print(f"Saved metrics to: {csv_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("AGGREGATE METRICS (WHITEBOX METHODS)")
    print("="*80)
    print(metrics_df.to_string(index=False))
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Run RAG whitebox attribution experiments')
    parser.add_argument('--dataset', type=str, default='data/complementary.csv',
                       help='Path to dataset file')
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.2-1B',
                       help='HuggingFace model name')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu), auto-detect if None')
    parser.add_argument('--max-queries', type=int, default=None,
                       help='Maximum number of queries to process')
    parser.add_argument('--methods', nargs='+', 
                       choices=['gradient', 'integrated_gradients', 'attention'],
                       default=None,
                       help='Whitebox attribution methods to use')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory for results')
    parser.add_argument('--ig-baseline', type=str, default='zero',
                       choices=['zero', 'pad'],
                       help='Baseline type for integrated gradients')
    parser.add_argument('--ig-steps', type=int, default=50,
                       help='Number of steps for integrated gradients')
    parser.add_argument('--attn-implementation', type=str, default=None,
                       choices=['eager', 'sdpa', 'flash_attention_2'],
                       help="Attention implementation. Use 'eager' for attention attribution method.")
    
    args = parser.parse_args()
    
    # Auto-select eager attention if 'attention' method is requested (or will be used by default)
    attn_impl = args.attn_implementation
    # If methods not specified, default includes 'attention'; if specified, check if 'attention' is in list
    uses_attention = (args.methods is None) or ('attention' in args.methods)
    print(f"DEBUG: args.methods={args.methods}, uses_attention={uses_attention}, attn_impl before={attn_impl}")
    if uses_attention and attn_impl is None:
        print("Note: 'attention' method requires eager attention. Setting --attn-implementation=eager")
        attn_impl = 'eager'
    print(f"DEBUG: attn_impl after={attn_impl}")
    
    # Run experiment
    results = run_whitebox_attribution_experiment(
        dataset_path=args.dataset,
        model_name=args.model,
        device=args.device,
        max_queries=args.max_queries,
        methods=args.methods,
        ig_baseline_type=args.ig_baseline,
        ig_n_steps=args.ig_steps,
        attn_implementation=attn_impl
    )
    
    # Save results
    save_results(results, output_dir=args.output_dir)


if __name__ == '__main__':
    main()

