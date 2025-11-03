import json
from typing import List, Tuple, Dict

EVAL_PATH = "outputs/eval_results.jsonl"

def load_rollouts(path: str) -> List[Dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            records.append(rec)
    return records

def compute_trajectory_metrics(records: List[Dict]) -> Tuple[float, float, float]:
    TP = FP = FN = 0
    for rec in records:
        if rec.get("pred_success") and rec.get("gold_success"):
            TP += 1
        if rec.get("pred_success") and not rec.get("gold_success"):
            FP += 1
        if not rec.get("pred_success") and rec.get("gold_success"):
            FN += 1
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

def compute_action_metrics(records: List[Dict]) -> Tuple[float, float, float]:
    TP = FP = FN = 0
    for rec in records:
        gold = rec.get("gold_actions", [])
        pred = rec.get("pred_actions", [])
        L = min(len(gold), len(pred))
        for i in range(L):
            if pred[i] == gold[i]:
                TP += 1
            else:
                FP += 1
                FN += 1
        if len(pred) > L:
            FP += (len(pred) - L)
        if len(gold) > L:
            FN += (len(gold) - L)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

def compute_simple_metrics(records: List[Dict]) -> float:
    matches = sum(1 for r in records if r.get("match", False))
    accuracy = matches / len(records) if len(records) > 0 else 0.0
    return accuracy

def print_detailed_results(records: List[Dict]):
    print("\n" + "="*70)
    print("DETAILED EVALUATION RESULTS")
    print("="*70)
    
    for rec in records:
        task_id = rec.get("task_id", "unknown")
        match = rec.get("match", False)
        
        print(f"\n{task_id}: {'✅ MATCH' if match else '❌ NO MATCH'}")
        print(f"User Message: {rec.get('user_message', '')[:100]}...")
        print(f"Expected: {rec.get('expected_response', '')[:100]}...")
        print(f"Predicted: {rec.get('predicted_response', '')[:100]}...")

def main():
    records = load_rollouts(EVAL_PATH)
    
    print("="*70)
    print("EVALUATION METRICS")
    print("="*70)
    
    if "match" in records[0]:
        accuracy = compute_simple_metrics(records)
        print(f"\nAccuracy: {accuracy:.4f} ({sum(r.get('match', False) for r in records)}/{len(records)} correct)")
    
    if "gold_success" in records[0] and "pred_success" in records[0]:
        traj_p, traj_r, traj_f1 = compute_trajectory_metrics(records)
        print("\n== Trajectory-level ==")
        print(f"Precision: {traj_p:.4f} | Recall: {traj_r:.4f} | F1: {traj_f1:.4f}")
    
    if "gold_actions" in records[0] and "pred_actions" in records[0]:
        action_p, action_r, action_f1 = compute_action_metrics(records)
        print("\n== Action-level ==")
        print(f"Precision: {action_p:.4f} | Recall: {action_r:.4f} | F1: {action_f1:.4f}")
    
    print_detailed_results(records)

if __name__ == "__main__":
    main()
