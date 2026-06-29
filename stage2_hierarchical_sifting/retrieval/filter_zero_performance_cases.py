import argparse
import json
import os
from typing import Any, Dict, Tuple


def load_zero_perf_decisions(results_json_path: str) -> Dict[int, str]:
    with open(results_json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    detailed = obj.get("detailed_results", [])
    decisions: Dict[int, str] = {}

    for entry in detailed:
        try:
            test_id = int(entry.get("test_id"))
        except Exception:
            continue

        result_perf = entry.get("result_performance", None)
        if result_perf != 0.0:
            continue

        valid_reps = entry.get("valid_representatives", []) or []
        rep_cnt = len(valid_reps)

        aggregated_perfs = entry.get("aggregated_performances", None)
        selected_model = entry.get("selected_model", None)

        rule1 = False
        if isinstance(aggregated_perfs, dict) and rep_cnt > 0:
            bad_models = 0
            for _, v in aggregated_perfs.items():
                try:
                    if float(v) + 1e-12 < float(rep_cnt):
                        bad_models += 1
                except Exception:
                    continue

            selected_bad = False
            if selected_model is not None and selected_model in aggregated_perfs:
                try:
                    selected_bad = float(aggregated_perfs[selected_model]) + 1e-12 < float(rep_cnt)
                except Exception:
                    selected_bad = False

            if bad_models >= 3 or selected_bad:
                rule1 = True

        uq_perfs = entry.get("user_query_performances", None)
        rule2 = False
        if isinstance(uq_perfs, dict):
            zero_cnt = 0
            for _, v in uq_perfs.items():
                try:
                    if float(v) == 0.0:
                        zero_cnt += 1
                except Exception:
                    continue
            if zero_cnt >= 4:
                rule2 = True

        if rule1:
            decisions[test_id] = "rule1"
        elif rule2:
            decisions[test_id] = "rule2"
        else:
            decisions[test_id] = "rule3"

    return decisions


def set_valid_representatives_empty(record: Dict[str, Any]) -> Tuple[bool, str]:
    if isinstance(record.get("judgment"), dict) and "valid_representatives" in record["judgment"]:
        record["judgment"]["valid_representatives"] = []
        return True, "judgment.valid_representatives"

    if "valid_representatives" in record:
        record["valid_representatives"] = []
        return True, "valid_representatives"

    if isinstance(record.get("judgment"), dict):
        record["judgment"]["valid_representatives"] = []
        return True, "judgment.valid_representatives(created)"

    record["valid_representatives"] = []
    return True, "valid_representatives(created)"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_json",
        type=str,
        default=os.path.join(os.path.dirname(__file__), 
            "Dataset",
            "output",
            "similarity_dataset",
            "llm_judge_v2",
            "test",
            "Test_testset_results_v1.json"),
    )
    parser.add_argument(
        "--input_jsonl",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__),
            "Dataset",
            "output",
            "similarity_dataset",
            "llm_judge_v2",
            "test",
            "similarity_judge_results_test_v1.jsonl",
        ),
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__),
            "Dataset",
            "output",
            "similarity_dataset",
            "llm_judge_v2",
            "test",
            "similarity_judge_results_v1.filtered.jsonl",
        ),
    )
    args = parser.parse_args()

    decisions = load_zero_perf_decisions(args.results_json)

    cnt_rule1 = 0
    cnt_rule2 = 0
    cnt_rule3 = 0
    cnt_total_read = 0
    cnt_total_written = 0
    cnt_modified = 0
    cnt_deleted = 0
    cnt_targeted_seen = 0

    os.makedirs(os.path.dirname(args.output_jsonl), exist_ok=True)

    with open(args.input_jsonl, "r", encoding="utf-8") as fin, open(
        args.output_jsonl, "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            cnt_total_read += 1
            try:
                rec = json.loads(line)
            except Exception:
                continue

            test_id = rec.get("test_id", None)
            try:
                test_id_int = int(test_id) if test_id is not None else None
            except Exception:
                test_id_int = None

            decision = decisions.get(test_id_int, None)
            if decision is None:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                cnt_total_written += 1
                continue

            cnt_targeted_seen += 1

            if decision == "rule1":
                cnt_rule1 += 1
                ok, _ = set_valid_representatives_empty(rec)
                if ok:
                    cnt_modified += 1
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                cnt_total_written += 1

            elif decision == "rule2":
                cnt_rule2 += 1
                ok, _ = set_valid_representatives_empty(rec)
                if ok:
                    cnt_modified += 1
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                cnt_total_written += 1

            else:
                cnt_rule3 += 1
                cnt_deleted += 1
                continue

    targeted_total = len(decisions)

    print("==== Filter Summary ====")
    print(f"results.json zero-perf samples (decision targets): {targeted_total}")
    print(f"targets actually found in input jsonl: {cnt_targeted_seen}")
    print("")
    print(f"rule1 (set valid_representatives=[]): {cnt_rule1}")
    print(f"rule2 (set valid_representatives=[]): {cnt_rule2}")
    print(f"rule3 (deleted): {cnt_rule3}")
    print("")
    print(f"input lines read: {cnt_total_read}")
    print(f"output lines written: {cnt_total_written}")
    print(f"modified lines written: {cnt_modified}")
    print(f"deleted lines (not written): {cnt_deleted}")
    print(f"output file: {args.output_jsonl}")


if __name__ == "__main__":
    main()
