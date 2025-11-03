import json
from typing import List, Dict

EVAL_PATH = "outputs/eval_results.jsonl"

def load_results(path: str) -> List[Dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            records.append(rec)
    return records

def compute_code_metrics(records: List[Dict]) -> Dict[str, float]:
    total = len(records)
    with_code = sum(1 for r in records if r.get("has_code", False))
    
    avg_gen_length = sum(len(r.get("predicted_response", "")) for r in records) / total if total > 0 else 0
    avg_ref_length = sum(len(r.get("expected_response", "")) for r in records) / total if total > 0 else 0
    
    return {
        "code_generation_rate": with_code / total if total > 0 else 0.0,
        "avg_generated_length": avg_gen_length,
        "avg_reference_length": avg_ref_length
    }

def print_detailed_results(records: List[Dict]):
    print("\n" + "="*70)
    print("DETAILED EVALUATION RESULTS")
    print("="*70)
    
    for rec in records:
        task_id = rec.get("task_id", "unknown")
        has_code = rec.get("has_code", False)
        
        print(f"\n{task_id}: {'✅ HAS CODE' if has_code else '❌ NO CODE'}")
        print(f"User Message: {rec.get('user_message', '')[:100]}...")
        print(f"System Prompt: {rec.get('system_prompt', '')[:100]}...")
        print(f"Expected length: {len(rec.get('expected_response', ''))} chars")
        print(f"Generated length: {len(rec.get('predicted_response', ''))} chars")
        print(f"Generated preview:\n{rec.get('predicted_response', '')[:200]}...")

def main():
    records = load_results(EVAL_PATH)
    
    print("="*70)
    print("REACT CODE GENERATION METRICS")
    print("="*70)
    
    metrics = compute_code_metrics(records)
    
    print(f"\nCode Generation Rate: {metrics['code_generation_rate']:.2%}")
    print(f"Avg Generated Length: {metrics['avg_generated_length']:.0f} chars")
    print(f"Avg Reference Length: {metrics['avg_reference_length']:.0f} chars")
    print(f"Total Examples: {len(records)}")
    
    print_detailed_results(records)

if __name__ == "__main__":
    main()
