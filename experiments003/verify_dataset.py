import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import json
import math
import hashlib
import statistics
from collections import Counter

def verify_dataset():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    qa_path = os.path.join(base_dir, "QAmdataset.json")
    sim_path = os.path.join(base_dir, "experiment_results", "datasets", "simulated_dataset.json")

    def file_info(p):
        if not os.path.exists(p):
            return {"exists": False, "path": p}
        size = os.path.getsize(p)
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return {"exists": True, "path": p, "size_bytes": size, "sha256": h.hexdigest()}

    files_summary = {
        "QAmdataset.json": file_info(qa_path),
        "simulated_dataset.json": file_info(sim_path),
    }

    with open(qa_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_records = len(data)
    empty_inputs = sum(1 for item in data if not item.get("input", "").strip())
    empty_outputs = sum(1 for item in data if not item.get("output", "").strip())

    # Sample first 10,000 records and full-corpus statistics
    input_word_counts = []
    output_word_counts = []
    input_char_counts = []
    output_char_counts = []
    unique_inputs = set()
    unique_outputs = set()

    for item in data:
        inp = item.get("input", "")
        out = item.get("output", "")
        unique_inputs.add(inp)
        unique_outputs.add(out)
        input_char_counts.append(len(inp))
        output_char_counts.append(len(out))
        input_word_counts.append(len(inp.split()))
        output_word_counts.append(len(out.split()))

    # Verify preprocessing for N=80, N=100, N=200, N=500, N=1422 as used in scripts/manuscript
    subset_stats = {}
    for n in [80, 100, 200, 500, 1422]:
        sub = data[:n]
        vocab_counter = Counter()
        processed_sequences = []
        for item in sub:
            text = f"{item.get('input', '')} {item.get('output', '')}"
            words = [w.strip(".,?!:;()\"'") for w in text.split()]
            words = [w for w in words if w]
            vocab_counter.update(words)
            processed_sequences.append(words)
        sorted_vocab = [word for word, freq in vocab_counter.most_common()]
        vocab_size = len(sorted_vocab)
        word2id = {word: idx + 1 for idx, word in enumerate(sorted_vocab)}
        id2complexity = {word2id[word]: rank / vocab_size for rank, word in enumerate(sorted_vocab, start=1)}
        flat_tokens = []
        flat_complexities = []
        for seq in processed_sequences:
            for w in seq:
                tid = word2id[w]
                flat_tokens.append(tid)
                flat_complexities.append(id2complexity[tid])
        eval_tokens = flat_tokens[:n]
        eval_comps = flat_complexities[:n]
        top10_freq = sum(freq for _, freq in vocab_counter.most_common(10)) / max(1, len(flat_tokens))
        subset_stats[f"num_samples_{n}"] = {
            "qa_records_loaded": len(sub),
            "total_extracted_words_in_records": len(flat_tokens),
            "vocab_size_in_records": vocab_size,
            "evaluated_token_stream_length": len(eval_tokens),
            "evaluated_next_token_transitions": len(eval_tokens) - 1,
            "mean_token_complexity": round(statistics.mean(eval_comps), 4),
            "std_token_complexity": round(statistics.stdev(eval_comps), 4),
            "top_10_words_frequency_share": round(top10_freq, 4),
            "unique_tokens_in_eval_window": len(set(eval_tokens)),
        }

    # Simulated dataset check
    with open(sim_path, "r", encoding="utf-8") as f:
        sim_data = json.load(f)

    summary = {
        "files": files_summary,
        "corpus_statistics": {
            "total_qa_pairs": total_records,
            "unique_inputs": len(unique_inputs),
            "unique_outputs": len(unique_outputs),
            "empty_inputs": empty_inputs,
            "empty_outputs": empty_outputs,
            "mean_input_words": round(statistics.mean(input_word_counts), 2),
            "mean_output_words": round(statistics.mean(output_word_counts), 2),
            "mean_input_chars": round(statistics.mean(input_char_counts), 2),
            "mean_output_chars": round(statistics.mean(output_char_counts), 2),
        },
        "pipeline_subset_verification": subset_stats,
        "simulated_dataset_statistics": {
            "num_entries": len(sim_data),
            "keys": list(sim_data[0].keys()) if sim_data else [],
            "max_token_id": max(x["token_id"] for x in sim_data) if sim_data else 0,
            "min_complexity": min(x["complexity"] for x in sim_data) if sim_data else 0,
            "max_complexity": max(x["complexity"] for x in sim_data) if sim_data else 0,
        }
    }

    out_dir = os.path.join(base_dir, "experiment_results_verified")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "dataset_verification_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    verify_dataset()

