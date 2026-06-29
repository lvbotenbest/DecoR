import argparse
import json
from pathlib import Path


INSTRUCTION = (
    'You are a Capability Decomposition Engine. Your task is to decompose the user '
    'query into its capability-space representation C(q) = {S, K, D, F}. Follow all '
    'rules strictly and output JSON only.\n'
    '\n'
    '---\n'
    '\n'
    '1. Skill Set (S):\n'
    'Identify the skills required to answer the query.\n'
    '- You may freely generate categories.\n'
    '- Examples: reasoning, logical inference, mathematics, coding, translation, '
    'writing, information extraction, multi-step planning, role-playing, style '
    'imitation, summarization.\n'
    '- Output as a list.\n'
    'Also output "S_reason": one sentence explaining why these skills are required.\n'
    '\n'
    '2. Knowledge Domain (K):\n'
    'Identify the knowledge domains needed for the query.\n'
    '- Freely generated categories; no fixed list.\n'
    '- Examples: general knowledge, medicine, law, finance, computer science, '
    'physics, ACG, history, philosophy.\n'
    '- If no specific knowledge is required, output "none".\n'
    '- Output as a list.\n'
    'Also output "K_reason": one sentence explaining why these domains are needed.\n'
    '\n'
    '3. Difficulty / Instruction Complexity (D):\n'
    'Choose exactly one:\n'
    '- D0 (Trivial): almost no reasoning; direct/simple request.\n'
    '- D1 (Simple): mild understanding; single goal; light reasoning.\n'
    '- D2 (Moderate): multiple requirements or multi-step tasks; needs '
    'organization/judgment.\n'
    '- D3 (Hard): complex tasks requiring deep reasoning, planning, abstraction, '
    'or structured logic.\n'
    'Also output "D_reason": one sentence explaining why this difficulty level '
    'matches the query.\n'
    '---\n'
    '\n'
    'IMPORTANT RULES:\n'
    '- Output MUST be valid pure JSON.\n'
    '- Do NOT include markdown code fences such as ```json or ``` anywhere.\n'
    '\n'
    '\n'
    'Output Format (STRICT):\n'
    '\n'
    'Return ONLY the following JSON structure:\n'
    '\n'
    '{\n'
    '  "S": [...],\n'
    '  "S_reason": "...",\n'
    '  "K": [...],\n'
    '  "K_reason": "...",\n'
    '  "D": "D0 or D1 or D2 or D3",\n'
    '  "D_reason": "...",\n'
    '}\n'
    '\n'
    'No explanations. No extra text. Only valid JSON.\n'
)


def convert_file(input_path: Path, output_path: Path) -> None:
    """Convert query_decomposition_reason_common_train.jsonl to instruction/input/output format."""
    if not input_path.exists():
        raise FileNotFoundError(f'Input file not found: {input_path}')

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open('r', encoding='utf-8') as src, output_path.open(
        'w', encoding='utf-8'
    ) as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            prompt = record.get('prompt')
            output = record.get('decomposition')
            if prompt is None or output is None:
                continue

            example = {
                'instruction': INSTRUCTION,
                'input': prompt,
                'output': output,
            }
            dst.write(json.dumps(example, ensure_ascii=False) + '\n')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Convert query decomposition JSONL into instruction/input/output format.'
    )
    parser.add_argument(
        '--input',
        default='query_decomposition_reason_common_train.jsonl',
        help='Path to source JSONL file.',
    )
    parser.add_argument(
        '--output',
        default='capability_decomposition_train.jsonl',
        help='Path to write the converted JSONL file.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_file(Path(args.input), Path(args.output))


if __name__ == '__main__':
    main()
