"""
Model Selection Script based on Similarity Judge Results

This script selects models based on similarity judge results and calculates
performance/cost metrics using a balance function.
"""

import json
import argparse
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


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

# Reverse mapping: real name to label
REVERSE_MODEL_MAPPING = {v: k for k, v in MODEL_MAPPING.items()}


def load_jsonl(file_path: str) -> List[dict]:
    """Load a JSONL file and return a list of dictionaries."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def build_corpus_index(corpus_data: List[dict]) -> Dict[int, dict]:
    """Build an index from corpus ID to corpus entry."""
    return {entry['id']: entry for entry in corpus_data}


def balance_function(performance: float, cost: float, 
                     alpha: float = 1.0, beta: float = 0.0) -> float:
    """
    Balance function to trade off between performance and cost.
    
    Score = alpha * performance - beta * cost
    
    Args:
        performance: Total/aggregated performance score
        cost: Total/aggregated cost
        alpha: Weight for performance (default 1.0)
        beta: Weight for cost penalty (default 0.0, meaning only consider performance)
    
    Returns:
        Balanced score (higher is better)
    """
    return alpha * performance - beta * cost


def get_model_real_name(label: str) -> str:
    """Convert model label (A, B, C...) to real model name."""
    return MODEL_MAPPING.get(label, label)


def get_model_label(real_name: str) -> str:
    """Convert real model name to label (A, B, C...)."""
    return REVERSE_MODEL_MAPPING.get(real_name, real_name)


def aggregate_model_scores(valid_representative_ids: List[int], 
                           corpus_index: Dict[int, dict]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Aggregate performance and cost scores for each model based on valid representative queries.
    
    Args:
        valid_representative_ids: List of query IDs that are valid representatives
        corpus_index: Index mapping query ID to corpus entry
    
    Returns:
        Tuple of (aggregated_performances, aggregated_costs) dictionaries with real model names as keys
    """
    aggregated_performances = defaultdict(float)
    aggregated_costs = defaultdict(float)
    
    for query_id in valid_representative_ids:
        if query_id not in corpus_index:
            print(f"Warning: Query ID {query_id} not found in corpus index")
            continue
        
        entry = corpus_index[query_id]
        performances = entry.get('performances', {})
        costs = entry.get('costs', {})
        
        for label, perf in performances.items():
            real_name = get_model_real_name(label)
            aggregated_performances[real_name] += perf
        
        for label, cost in costs.items():
            real_name = get_model_real_name(label)
            aggregated_costs[real_name] += cost
    
    return dict(aggregated_performances), dict(aggregated_costs)


def select_best_model(aggregated_performances: Dict[str, float],
                      aggregated_costs: Dict[str, float],
                      alpha: float = 1.0,
                      beta: float = 0.0) -> str:
    """
    Select the best model based on the balance function.
    
    Args:
        aggregated_performances: Dict of model name -> total performance
        aggregated_costs: Dict of model name -> total cost
        alpha: Weight for performance
        beta: Weight for cost penalty
    
    Returns:
        Name of the best model
    """
    best_model = None
    best_score = float('-inf')
    
    for model_name in aggregated_performances.keys():
        perf = aggregated_performances.get(model_name, 0)
        cost = aggregated_costs.get(model_name, 0)
        score = balance_function(perf, cost, alpha, beta)
        
        if score > best_score:
            best_score = score
            best_model = model_name
    
    return best_model


def get_user_query_result(test_id: int, selected_model: str, 
                          user_query_index: Dict[int, dict]) -> Tuple[float, float]:
    """
    Get the performance and cost of the selected model for a user query.
    
    Args:
        test_id: The test ID of the user query
        selected_model: The name of the selected model
        user_query_index: Index mapping test_id to user query entry
    
    Returns:
        Tuple of (performance, cost) for the selected model
    """
    if test_id not in user_query_index:
        print(f"Warning: Test ID {test_id} not found in user query index")
        return 0.0, 0.0
    
    entry = user_query_index[test_id]
    model_label = get_model_label(selected_model)
    
    performances = entry.get('performances', {})
    costs = entry.get('costs', {})
    
    performance = performances.get(model_label, 0.0)
    cost = costs.get(model_label, 0.0)
    
    return performance, cost


def process_judge_results(judge_file: str,
                          history_corpus_file: str,
                          user_query_full_file: str,
                          alpha: float = 1.0,
                          beta: float = 0.0,
                          default_model: str = "google_gemma-3-12b-it",
                          output_file: str = None) -> Dict:
    """
    Main processing function.
    
    Args:
        judge_file: Path to the similarity judge results file
        history_corpus_file: Path to the history corpus file (full_query_decomposition_QWEN.jsonl)
        user_query_full_file: Path to the user query full file
        alpha: Weight for performance in balance function
        beta: Weight for cost penalty in balance function
        default_model: Default model to use when valid_representatives is empty
        output_file: Optional path to save detailed results
    
    Returns:
        Dictionary containing evaluation metrics
    """
    print(f"Loading judge file: {judge_file}")
    judge_data = load_jsonl(judge_file)
    
    print(f"Loading history corpus file: {history_corpus_file}")
    history_corpus = load_jsonl(history_corpus_file)
    history_corpus_index = build_corpus_index(history_corpus)
    
    print(f"Loading user query full file: {user_query_full_file}")
    user_query_full = load_jsonl(user_query_full_file)
    user_query_index = {entry['id']: entry for entry in user_query_full}
    
    total_performance = 0.0
    total_cost = 0.0
    num_queries = 0
    num_empty_representatives = 0
    
    detailed_results = []
    
    for judge_entry in judge_data:
        test_id = judge_entry.get('test_id')
        user_query = judge_entry.get('user_query', '')
        
        # valid_representatives can be either at top level or nested in judgment
        judgment = judge_entry.get('judgment', {})
        if isinstance(judgment, dict):
            valid_representatives = judgment.get('valid_representatives', [])
        else:
            valid_representatives = judge_entry.get('valid_representatives', [])
        retrieved = judge_entry.get('retrieved', [])
        
        result_entry = {
            'test_id': test_id,
            'user_query': user_query[:100] + '...' if len(user_query) > 100 else user_query,
            'valid_representatives': valid_representatives,
        }
        
        if not valid_representatives:
            # Use default model
            selected_model = default_model
            result_entry['selection_method'] = 'default'
            result_entry['selected_model'] = selected_model
            num_empty_representatives += 1
        else:
            # Get IDs of valid representative queries
            label_to_id = {item['label']: item['id'] for item in retrieved}
            valid_ids = [label_to_id[label] for label in valid_representatives if label in label_to_id]
            
            if not valid_ids:
                # Fallback to default model if no valid IDs found
                selected_model = default_model
                result_entry['selection_method'] = 'default_fallback'
                result_entry['selected_model'] = selected_model
            else:
                # Aggregate scores from valid representative queries
                agg_perfs, agg_costs = aggregate_model_scores(valid_ids, history_corpus_index)
                # print(agg_perfs, agg_costs)
                # Select best model using balance function
                selected_model = select_best_model(agg_perfs, agg_costs, alpha, beta)
                
                result_entry['selection_method'] = 'similarity_based'
                result_entry['selected_model'] = selected_model
                result_entry['aggregated_performances'] = agg_perfs
                result_entry['aggregated_costs'] = agg_costs
        # print(selected_model)
        # Get performance and cost for the selected model on the user query
        performance, cost = get_user_query_result(test_id, selected_model, user_query_index)
        
        # Add performances for all models on this user query (but not costs)
        if test_id in user_query_index:
            uq_entry = user_query_index[test_id]
            uq_perfs = uq_entry.get('performances', {})
            user_query_performances = {}
            for label, score in uq_perfs.items():
                 real_name = get_model_real_name(label)
                 user_query_performances[real_name] = score
            result_entry['user_query_performances'] = user_query_performances

        result_entry['result_performance'] = performance
        result_entry['result_cost'] = cost
        
        total_performance += performance
        total_cost += cost
        num_queries += 1
        
        detailed_results.append(result_entry)
    

    # Calculate metrics
    avg_performance = total_performance / num_queries if num_queries > 0 else 0.0
    avg_cost = total_cost / num_queries if num_queries > 0 else 0.0
    
    metrics = {
        'num_queries': num_queries,
        'num_empty_representatives': num_empty_representatives,
        'average_performance': avg_performance,
        'total_cost': total_cost,
        'average_cost': avg_cost,
        'alpha': alpha,
        'beta': beta,
        'default_model': default_model
    }
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total queries processed: {num_queries}")
    print(f"Queries with empty representatives (used default): {num_empty_representatives}")
    print(f"Average Performance: {avg_performance:.4f}")
    print(f"Total Cost: {total_cost:.6f}")
    print(f"Average Cost: {avg_cost:.8f}")
    print(f"Alpha (performance weight): {alpha}")
    print(f"Beta (cost weight): {beta}")
    print(f"Default Model: {default_model}")
    print("=" * 60)
    
    # Save detailed results if output file specified
    if output_file:
        output_data = {
            'metrics': metrics,
            'detailed_results': detailed_results
        }
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\nDetailed results saved to: {output_file}")
    
    return {
        'metrics': metrics,
        'detailed_results': detailed_results
    }


def main():
    parser = argparse.ArgumentParser(
        description='Model selection based on similarity judge results'
    )
    parser.add_argument(
        '--judge_file', '-j',
        type=str,
        required=True,
        help='Path to the similarity judge results file (JSONL)'
    )
    parser.add_argument(
        '--history_corpus', '-hc',
        type=str,
        required=True,
        help='Path to the history corpus file (full_query_decomposition_QWEN.jsonl)'
    )
    parser.add_argument(
        '--user_query_full', '-uq',
        type=str,
        required=True,
        help='Path to the user query full file (with performances and costs)'
    )
    parser.add_argument(
        '--alpha', '-a',
        type=float,
        default=1.0,
        help='Weight for performance in balance function (default: 1.0)'
    )
    parser.add_argument(
        '--beta', '-b',
        type=float,
        default=0.0,
        help='Weight for cost penalty in balance function (default: 0.0)'
    )
    parser.add_argument(
        '--default_model', '-dm',
        type=str,
        default='google_gemma-3-12b-it',
        help='Default model to use when valid_representatives is empty'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Path to save detailed results (JSON)'
    )
    
    args = parser.parse_args()
    
    process_judge_results(
        judge_file=args.judge_file,
        history_corpus_file=args.history_corpus,
        user_query_full_file=args.user_query_full,
        alpha=args.alpha,
        beta=args.beta,
        default_model=args.default_model,
        output_file=args.output
    )


if __name__ == '__main__':
    main()
