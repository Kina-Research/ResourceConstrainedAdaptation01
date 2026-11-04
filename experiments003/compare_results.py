import os
import json
import pandas as pd
import numpy as np

def compare_all():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    orig_dir = os.path.join(base_dir, "experiment_results")
    ver_dir = os.path.join(base_dir, "experiment_results_verified")

    report = {
        "comparison_summary": {},
        "discrepancy_investigation": []
    }

    # 1. Compare 4-Pipeline Comparison Table (Original vs Verified 200-sample run)
    orig_pipe = pd.read_csv(os.path.join(orig_dir, "pipeline_comparison_table.csv"))
    ver200_pipe_path = os.path.join(ver_dir, "run_200_samples", "pipeline_comparison_table.csv")
    if os.path.exists(ver200_pipe_path):
        ver200_pipe = pd.read_csv(ver200_pipe_path)
        diffs = {}
        for col in ["Mean_Perplexity", "CI_Low", "CI_High", "Active_Params_%", "Joules_per_Token", "Accuracy_%", "BLEU1", "CharF1"]:
            max_abs_diff = float(np.max(np.abs(orig_pipe[col] - ver200_pipe[col])))
            diffs[col] = {
                "max_abs_diff": max_abs_diff,
                "original_values": orig_pipe[col].tolist(),
                "verified_200_values": ver200_pipe[col].tolist(),
                "exact_match": bool(max_abs_diff < 1e-4)
            }
        report["comparison_summary"]["pipeline_comparison_table_200_samples"] = diffs

    # 2. Compare Baselines (Original vs Verified 200-sample run)
    orig_bl = pd.read_csv(os.path.join(orig_dir, "baselines", "baseline_comparison.csv"))
    ver200_bl_path = os.path.join(ver_dir, "run_200_samples", "baselines", "baseline_comparison.csv")
    if os.path.exists(ver200_bl_path):
        ver200_bl = pd.read_csv(ver200_bl_path)
        report["comparison_summary"]["baseline_comparison_200_samples"] = {
            "original": orig_bl.to_dict(orient="records"),
            "verified_200": ver200_bl.to_dict(orient="records"),
            "max_ppl_diff": float(np.max(np.abs(orig_bl["Mean_PPL"] - ver200_bl["Mean_PPL"]))),
            "max_acc_diff": float(np.max(np.abs(orig_bl["Accuracy_%"] - ver200_bl["Accuracy_%"])))
        }

    # 3. Compare RQ1, RQ2, RQ3, and Budget Ablation (Original vs Verified 500-sample and 200-sample runs)
    ablation_files = [
        ("budget_ablation", os.path.join("compute_optimal", "budget_ablation.csv")),
        ("rq1_grid", os.path.join("rq1_ablation", "rq1_grid.csv")),
        ("rq2_threshold_sweep", os.path.join("rq2_ablation", "rq2_threshold_sweep.csv")),
        ("rq3_peft_comparison", os.path.join("rq3_ablation", "rq3_peft_comparison.csv")),
        ("grid_search_results", "grid_search_results.csv"),
    ]

    for name, rel_path in ablation_files:
        orig_f = os.path.join(orig_dir, rel_path)
        if not os.path.exists(orig_f):
            continue
        df_orig = pd.read_csv(orig_f)
        entry = {"original_rows": len(df_orig), "original": df_orig.to_dict(orient="records")}
        for run_name in ["run_200_samples", "run_500_samples", "run_100_samples"]:
            ver_f = os.path.join(ver_dir, run_name, rel_path)
            if os.path.exists(ver_f):
                df_ver = pd.read_csv(ver_f)
                num_cols = df_orig.select_dtypes(include=[np.number]).columns
                col_diffs = {}
                for c in num_cols:
                    if c in df_ver.columns and len(df_orig) == len(df_ver):
                        col_diffs[c] = float(np.max(np.abs(df_orig[c] - df_ver[c])))
                entry[run_name] = {
                    "max_numeric_diffs_vs_original": col_diffs,
                    "exact_match_with_original": all(v < 1e-4 for v in col_diffs.values()) if col_diffs else False,
                    "records": df_ver.to_dict(orient="records")
                }
        report["comparison_summary"][name] = entry

    out_path = os.path.join(ver_dir, "comparison_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved comparison report to {out_path}")

if __name__ == "__main__":
    compare_all()

