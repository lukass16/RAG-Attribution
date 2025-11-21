#!/usr/bin/env python3
"""
Experiment script for RAG Source Attribution Analysis

This script runs attribution experiments across multiple queries and datasets,
computes evaluation metrics, and saves results to files.
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


def run_attribution_experiment(
    dataset_path: str,
    model_name: str = "meta-llama/Llama-3.2-1B",
    device: Optional[str] = None,
    max_queries: Optional[int] = None,
    methods: Optional[List[str]] = None
) -> Dict:
    """
    Run attribution experiment on a dataset.
    
    Args:
        dataset_path: Path to dataset CSV/JSON file
        model_name: HuggingFace model name
        device: Device to use ('cuda', 'cpu', or None for auto)
        max_queries: Maximum number of queries to process (None = all)
        methods: List of attribution methods to use (None = all)
    
    Returns:
        Dictionary containing results and metrics
    """
    print(f"\n{'='*80}")
    print(f"Running experiment on: {dataset_path}")
    print(f"{'='*80}\n")
    
    # Initialize RAG system
    print("Initializing RAG system...")
    rag = RAGSystem(model_name=model_name, device=device)
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
    for i, item in enumerate(dataset):
        print(f"  Processing query {i+1}/{len(dataset)}...", end='\r')
        rtarget = rag.generate_target_response(
            question=item['question'],
            all_documents=item['documents'],
            max_new_tokens=50
        )
        target_responses.append(rtarget)
    print(f"\nGenerated {len(target_responses)} target responses\n")
    
    # Default methods if not specified
    if methods is None:
        methods = ['leave_one_out', 'permutation_shapley', 'monte_carlo_shapley']
    
    # Results storage
    all_results = []
    
    # Process each query
    for query_idx, item in enumerate(dataset):
        print(f"\nProcessing query {query_idx + 1}/{len(dataset)}")
        print(f"Question: {item['question'][:80]}...")
        
        target_response = target_responses[query_idx]
        doc_ids = item['document_ids']
        n_docs = len(item['documents'])
        
        # Create utility function
        def utility_function(document_subset: List[str]) -> float:
            return rag.compute_utility(
                question=item['question'],
                document_subset=document_subset,
                target_response=target_response
            )
        
        query_results = {
            'query_idx': query_idx,
            'question': item['question'],
            'expected_answer': item.get('answer', ''),
            'target_response': target_response,
            'n_documents': n_docs,
            'document_ids': doc_ids,
            'attributions': {}
        }
        
        # Compute attributions for each method
        for method_name in methods:
            try:
                if method_name == 'leave_one_out':
                    attributions = attr.leave_one_out(item['documents'], utility_function)
                
                elif method_name == 'permutation_shapley':
                    attributions = attr.permutation_shapley(
                        item['documents'], utility_function, num_permutations=50
                    )
                
                elif method_name == 'monte_carlo_shapley':
                    if n_docs <= 10:
                        attributions = attr.monte_carlo_shapley(
                            item['documents'], utility_function, num_samples=64
                        )
                    else:
                        print(f"    Skipping Monte Carlo Shapley - too many documents ({n_docs})")
                        continue
                
                elif method_name == 'kernel_shap':
                    if n_docs <= 10:
                        attributions = attr.kernel_shap(
                            item['documents'], utility_function, num_samples=64
                        )
                    else:
                        print(f"    Skipping Kernel SHAP - too many documents ({n_docs})")
                        continue
                
                elif method_name == 'exact_shapley':
                    if n_docs <= 5:
                        attributions = attr.exact_shapley(item['documents'], utility_function)
                    else:
                        print(f"    Skipping Exact Shapley - too many documents ({n_docs})")
                        continue
                
                else:
                    print(f"    Unknown method: {method_name}")
                    continue
                
                # Convert attributions to list format (handle dict, array, or list)
                if isinstance(attributions, dict):
                    # Convert dict {index: value} to list [value_0, value_1, ...]
                    attributions_list = [attributions.get(i, 0.0) for i in range(len(item['documents']))]
                elif isinstance(attributions, np.ndarray):
                    attributions_list = attributions.tolist()
                else:
                    attributions_list = list(attributions)
                
                # Store attributions
                query_results['attributions'][method_name] = attributions_list
                
                # Use list for metric computation
                attributions_for_metrics = attributions_list
                
                # Compute metrics
                top_2_accuracy = compute_top_k_accuracy(attributions_for_metrics, doc_ids, k=2)
                ranks = compute_rank_of_correct_docs(attributions_for_metrics, doc_ids)
                
                query_results[f'{method_name}_top2_accuracy'] = top_2_accuracy
                query_results[f'{method_name}_rank_A'] = ranks.get('A', None)
                query_results[f'{method_name}_rank_B'] = ranks.get('B', None)
                
                print(f"    {method_name}: Top-2 accuracy = {top_2_accuracy}, Rank A = {ranks.get('A')}, Rank B = {ranks.get('B')}")
                
            except Exception as e:
                print(f"    Error computing {method_name}: {e}")
                query_results['attributions'][method_name] = None
        
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
    
    return {
        'dataset_path': dataset_path,
        'model_name': model_name,
        'timestamp': datetime.now().isoformat(),
        'n_queries': len(dataset),
        'methods': methods,
        'results': all_results,
        'aggregate_metrics': aggregate_metrics
    }


def save_results(results: Dict, output_dir: str = "results"):
    """Save experiment results to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = Path(results['dataset_path']).stem
    
    # Save full results as JSON
    json_path = os.path.join(output_dir, f"{dataset_name}_{timestamp}_full.json")
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
    csv_path = os.path.join(output_dir, f"{dataset_name}_{timestamp}_metrics.csv")
    metrics_df.to_csv(csv_path, index=False)
    print(f"Saved metrics to: {csv_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("AGGREGATE METRICS")
    print("="*80)
    print(metrics_df.to_string(index=False))
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Run RAG attribution experiments')
    parser.add_argument('--dataset', type=str, default='data/complementary.csv',
                       help='Path to dataset file')
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.2-1B',
                       help='HuggingFace model name')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu), auto-detect if None')
    parser.add_argument('--max-queries', type=int, default=None,
                       help='Maximum number of queries to process')
    parser.add_argument('--methods', nargs='+', 
                       choices=['leave_one_out', 'permutation_shapley', 
                               'monte_carlo_shapley', 'kernel_shap', 'exact_shapley'],
                       default=None,
                       help='Attribution methods to use')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Run experiment
    results = run_attribution_experiment(
        dataset_path=args.dataset,
        model_name=args.model,
        device=args.device,
        max_queries=args.max_queries,
        methods=args.methods
    )
    
    # Save results
    save_results(results, output_dir=args.output_dir)


if __name__ == '__main__':
    main()

