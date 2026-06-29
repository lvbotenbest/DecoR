import re
import json

def compute_score_llmroute(data_source, extra_info, solution_str, ground_truth, method="strict", format_score=0.0, score=1.0):
    def _parse_pred_list(text: str):
        if not text:
            return None
        idx = text.rfind("####")
        if idx == -1:
            return None
        tail = text[idx + 4 :].strip()

        # Try to locate a JSON list within the tail.
        # We intentionally take the first bracketed list to be robust to extra text.
        m = re.search(r"\[[\s\S]*?\]", tail)
        if not m:
            return None
        list_str = m.group(0)
        try:
            parsed = json.loads(list_str)
        except Exception:
            return None
        if not isinstance(parsed, list):
            return None

        out = []
        for x in parsed:
            if isinstance(x, str):
                s = x.strip()
                if s:
                    out.append(s)
        return out

    def _normalize_gt_list(gt):
        if gt is None:
            return []
        if isinstance(gt, (list, tuple, set)):
            out = []
            for x in gt:
                if isinstance(x, str):
                    s = x.strip()
                    if s:
                        out.append(s)
            return out
        if isinstance(gt, str):
            s = gt.strip()
            if not s or s in {"[]", "null", "None"}:
                return []
            # If dataloader serialized list as a string, try to parse.
            if s.startswith("[") and s.endswith("]"):
                for cand in (s, s.replace("'", '"')):
                    try:
                        parsed = json.loads(cand)
                    except Exception:
                        parsed = None
                    if isinstance(parsed, list):
                        return [
                            y.strip()
                            for y in parsed
                            if isinstance(y, str) and y.strip()
                        ]
            # Fallback: treat as a single label string.
            return [s]
        return []

    pred_list = _parse_pred_list(solution_str)
    if pred_list is None:
        return 0

    # ground_truth is expected to be List[str] from parquet schema.
    gt_list = _normalize_gt_list(ground_truth)

    P = set(pred_list)
    G = set(gt_list)

    # 1) exact match (including both empty)
    if P == G:
        return 6

    # 2) ground_truth empty but pred not empty: -2 per predicted item
    if len(G) == 0:
        return -2 * len(P)

    # 3) ground_truth non-empty but no hit at all: -6
    if len(P) > 0 and len(P.intersection(G)) == 0:
        return -6

    # 4) partial match: +w per hit, -w per false positive
    n = len(G)
    w = 6.0 / n
    hits = len(P.intersection(G))
    false_pos = len(P.difference(G))
    return w * (hits - false_pos)













