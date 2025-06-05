import datetime
import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np

# Provided by user
CURRENT_USER_LOGIN = "jrzkaminski"
CURRENT_UTC_DATETIME = str(datetime.datetime.now().timestamp())


def load_results(path: Path = Path("results_all.json")) -> List[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            results = json.load(f)
            if not isinstance(results, list):
                print(f"Error: Results file at {path} does not contain a JSON list.")
                return []
            return results
    except FileNotFoundError:
        print(f"Error: Results file not found at {path}")
        return []
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {path}")
        return []


def calculate_metrics(
    generated_calls_raw: List[Dict], reference_calls_raw: List[Dict]
) -> Dict[str, float]:
    metrics = {
        "InstAcc": 0.0,
        "ToolAcc": 0.0,
        "ArgAcc": 0.0,
        "Precision": 0.0,
        "Recall": 0.0,
        "F1": 0.0,
    }
    generated_calls = generated_calls_raw
    reference_calls = reference_calls_raw
    num_generated = len(generated_calls)
    num_reference = len(reference_calls)

    if num_reference == 0:
        return {k: 1.0 for k in metrics} if num_generated == 0 else metrics

    metrics["InstAcc"] = 1.0 if num_generated == num_reference else 0.0
    correct_tools = 0
    for i in range(min(num_generated, num_reference)):
        if generated_calls[i].get("tool") == reference_calls[i].get("tool"):
            correct_tools += 1
    metrics["ToolAcc"] = correct_tools / num_reference

    correct_args_count, total_reference_args_count = 0, 0
    for i in range(min(num_generated, num_reference)):
        if generated_calls[i].get("tool") == reference_calls[i].get("tool"):
            gen_param = generated_calls[i].get("param", {})
            ref_param = reference_calls[i].get("param", {})
            gen_args = set(gen_param.keys()) if isinstance(gen_param, dict) else set()
            ref_args = set(ref_param.keys()) if isinstance(ref_param, dict) else set()
            total_reference_args_count += len(ref_args)
            correct_args_count += len(gen_args.intersection(ref_args))
    metrics["ArgAcc"] = (
        correct_args_count / total_reference_args_count
        if total_reference_args_count > 0
        else 1.0
    )

    gen_tool_set = set(call.get("tool") for call in generated_calls if call.get("tool"))
    ref_tool_set = set(call.get("tool") for call in reference_calls if call.get("tool"))
    tp = len(gen_tool_set.intersection(ref_tool_set))
    fp = len(gen_tool_set - ref_tool_set)
    fn = len(ref_tool_set - gen_tool_set)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    metrics["Precision"] = precision
    metrics["Recall"] = recall
    metrics["F1"] = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return metrics


if __name__ == "__main__":
    results_file_path = Path("results_all.json")
    evaluation_output_path = Path("evaluation_summary.json")

    all_benchmark_results = load_results(results_file_path)
    if not all_benchmark_results:
        print("No results found. Exiting.")
        exit()

    print(
        f"--- Evaluating {len(all_benchmark_results)} Benchmark Results from {results_file_path} ---"
    )
    individual_metrics_data, all_metrics_lists = [], {
        "InstAcc": [],
        "ToolAcc": [],
        "ArgAcc": [],
        "Precision": [],
        "Recall": [],
        "F1": [],
        "RetriesSpent": [],  # Added RetriesSpent
    }
    total_prompt_tokens, total_completion_tokens, total_retries = 0, 0, 0

    for i, result_item in enumerate(all_benchmark_results):
        print(
            f"\n--- Result {i+1}/{len(all_benchmark_results)} (ID: {result_item.get('id', 'N/A')}) ---"
        )
        print(f"Question: {result_item.get('question', 'N/A')}")
        # ... (print generated/reference calls and error if any) ...

        prompt_tokens = result_item.get("prompt_tokens", 0)
        completion_tokens = result_item.get("completion_tokens", 0)
        retries_spent = result_item.get("retries_spent", 0)
        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens
        total_retries += retries_spent

        metrics = calculate_metrics(
            result_item.get("generated_calls", []),
            result_item.get("reference_calls", []),
        )
        metrics["RetriesSpent"] = float(
            retries_spent
        )  # Add retries to item metrics for consistency

        individual_metrics_data.append(
            {
                "id": result_item.get("id", f"result_{i+1}"),
                "question": result_item.get("question", "N/A"),
                "metrics": metrics,  # This now includes RetriesSpent
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "run_error": result_item.get("final_error"),
            }
        )

        for key, value in metrics.items():  # This will now include RetriesSpent
            if key in all_metrics_lists:
                all_metrics_lists[key].append(value)

        print("\nMetrics for this result:")
        for name, value in metrics.items():
            print(f"  {name}: {value:.4f}")

    aggregated_metrics_dict = {}
    print("\n--- Aggregate Metrics Across All Results ---")
    num_results = len(all_benchmark_results)
    if num_results > 0:
        print(f"Average metrics over {num_results} results:")
        for key, values_list in all_metrics_lists.items():
            average_value = np.mean(values_list) if values_list else 0.0
            agg_key_name = f"Avg_{key}"
            aggregated_metrics_dict[agg_key_name] = average_value
            print(f"  {agg_key_name}: {average_value:.4f}")

        aggregated_metrics_dict["Total_Prompt_Tokens"] = total_prompt_tokens
        aggregated_metrics_dict["Total_Completion_Tokens"] = total_completion_tokens
        aggregated_metrics_dict["Total_Retries"] = total_retries
        aggregated_metrics_dict["Avg_Prompt_Tokens_Per_Run"] = (
            total_prompt_tokens / num_results
        )
        aggregated_metrics_dict["Avg_Completion_Tokens_Per_Run"] = (
            total_completion_tokens / num_results
        )
        # Avg_RetriesSpent will be calculated by the loop above

        print(
            f"\nTotal Tokens (All Runs) - Prompt: {total_prompt_tokens}, Completion: {total_completion_tokens}"
        )
        print(f"Total Retries (All Runs): {total_retries}")
    else:
        print("\nNo metrics were calculated.")

    evaluation_output = {
        "evaluation_timestamp_utc": CURRENT_UTC_DATETIME,  # Use provided fixed value
        "evaluated_by_user": CURRENT_USER_LOGIN,  # Use provided fixed value
        "source_results_file": str(results_file_path.name),  # Just filename
        "total_benchmarks_processed": num_results,
        "aggregated_metrics": aggregated_metrics_dict,
        "individual_results_metrics": individual_metrics_data,
    }

    print(f"\n--- Saving evaluation summary to {evaluation_output_path} ---")
    try:
        with open(evaluation_output_path, "w", encoding="utf-8") as f:
            json.dump(evaluation_output, f, indent=2)
        print("Evaluation summary saved successfully.")
    except Exception as e:
        print(f"\nError saving evaluation summary to JSON: {e}")
