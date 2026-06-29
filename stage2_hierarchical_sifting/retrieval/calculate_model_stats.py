"""
Calculate per-model statistics from corpus file.

Computes average performance, total cost, and average cost for each model
across all queries in the corpus.
"""

import json
import argparse
from collections import defaultdict
from pathlib import Path


# Model label to real name mapping
MODEL_MAPPING = {
    "A": "deepseek-ai_DeepSeek-V3.1-Terminus",
    "B": "deepseek-ai_DeepSeek-V3.2-Exp",
    "C": "google_gemma-3-12b-it",
    "D": "google_gemma-3-27b-it",
    "E": "mistralai_Mistral-Small-3.2-24B-Instruct-2506",
    "F": "moonshotai_Kimi-K2-Instruct-0905",
    "G": "openai_gpt-oss-120b",
    "H": "Qwen_Qwen3-235B-A22B-Instruct-2507"
}


def load_jsonl(file_path: str) -> list:
    """Load a JSONL file and return a list of dictionaries."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def get_model_real_name(label: str) -> str:
    """Convert model label (A, B, C...) to real model name."""
    return MODEL_MAPPING.get(label, label)


def calculate_model_statistics(corpus_file: str, output_dir: str = None):
    """
    Calculate statistics for each model across all queries.
    
    Args:
        corpus_file: Path to the corpus JSONL file
        output_dir: Directory to save results (defaults to same as corpus file)
    """
    print(f"Loading corpus file: {corpus_file}")
    corpus_data = load_jsonl(corpus_file)
    
    if output_dir is None:
        output_dir = Path(corpus_file).parent
    else:
        output_dir = Path(output_dir)
    
    # Initialize accumulators
    total_performances = defaultdict(float)
    total_costs = defaultdict(float)
    query_counts = defaultdict(int)
    
    num_queries = len(corpus_data)
    
    # Aggregate statistics
    for entry in corpus_data:
        performances = entry.get('performances', {})
        costs = entry.get('costs', {})
        
        for label, perf in performances.items():
            real_name = get_model_real_name(label)
            total_performances[real_name] += perf
            query_counts[real_name] += 1
        
        for label, cost in costs.items():
            real_name = get_model_real_name(label)
            total_costs[real_name] += cost
    
    # Calculate averages and prepare results
    results = {}
    for model_name in total_performances.keys():
        count = query_counts[model_name]
        avg_perf = total_performances[model_name] / count if count > 0 else 0.0
        total_cost = total_costs[model_name]
        avg_cost = total_cost / count if count > 0 else 0.0
        
        results[model_name] = {
            "average_performance": avg_perf,
            "total_cost": total_cost,
            "average_cost": avg_cost,
            "num_queries": count
        }
    
    # Sort by average performance (descending)
    sorted_results = dict(sorted(results.items(), 
                                  key=lambda x: x[1]['average_performance'], 
                                  reverse=True))
    
    # Print results
    print("\n" + "=" * 80)
    print("MODEL STATISTICS")
    print("=" * 80)
    print(f"Total queries in corpus: {num_queries}")
    print("-" * 80)
    print(f"{'Model Name':<50} {'Avg Perf':>10} {'Total Cost':>12} {'Avg Cost':>12}")
    print("-" * 80)
    
    for model_name, stats in sorted_results.items():
        print(f"{model_name:<50} {stats['average_performance']:>10.4f} "
              f"{stats['total_cost']:>12.4f} {stats['average_cost']:>12.8f}")
    
    print("=" * 80)
    
    # Save results
    output_data = {
        "corpus_file": str(corpus_file),
        "total_queries": num_queries,
        "model_statistics": sorted_results
    }
    
    output_file = output_dir / "model_statistics.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return output_data


def main():
    parser = argparse.ArgumentParser(
        description='Calculate per-model statistics from corpus file'
    )
    parser.add_argument(
        '--corpus_file', '-c',
        type=str,
        default='Dataset/corpus/IID/full_query_decomposition_QWEN.jsonl',
        help='Path to the corpus JSONL file'
    )
    parser.add_argument(
        '--output_dir', '-o',
        type=str,
        default=None,
        help='Directory to save results (defaults to same as corpus file)'
    )
    
    args = parser.parse_args()
    
    calculate_model_statistics(
        corpus_file=args.corpus_file,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()
