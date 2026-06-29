import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Tuple

EXCLUDE_FIELDS = {
    'sample_id',
    'prompt',
    'eval_name',
    'oracle_model_to_route_to',
    'router_prompt',
    'oracle_chosen',
    'performances',
    'costs',
    'history_performance',
    'history_cost',
}


def iter_model_scores(entry: Dict) -> Iterable[Tuple[str, float]]:
    """Yield (model_name, score) pairs from a single JSONL entry."""
    for key, value in entry.items():
        if key in EXCLUDE_FIELDS or key.endswith('|total_cost'):
            continue
        if isinstance(value, (int, float)):
            yield key, float(value)


def collect_stats(jsonl_path: Path) -> Dict[str, Dict[str, float]]:
    """Return {eval_name: {model_name: accuracy}}."""
    stats = defaultdict(lambda: defaultdict(lambda: {'correct': 0.0, 'total': 0}))

    with jsonl_path.open('r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            eval_name = record.get('eval_name', 'UNKNOWN')

            for model_name, score in iter_model_scores(record):
                bucket = stats[eval_name][model_name]
                bucket['correct'] += score
                bucket['total'] += 1

    # Convert counts to accuracy values
    result: Dict[str, Dict[str, float]] = {}
    for dataset, model_stats in stats.items():
        dataset_result = {}
        for model_name, agg in model_stats.items():
            total = agg['total']
            dataset_result[model_name] = agg['correct'] / total if total else 0.0
        result[dataset] = dataset_result
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Compute per-eval per-model accuracy from a merged JSONL file.',
    )
    parser.add_argument(
        '--input',
        default='merged_query_decomposition_test.jsonl',
        help='Path to the merged JSONL file to analyze.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jsonl_path = Path(args.input)
    if not jsonl_path.exists():
        raise FileNotFoundError(f'Input file not found: {jsonl_path}')

    stats = collect_stats(jsonl_path)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
