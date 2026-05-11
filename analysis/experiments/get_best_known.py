"""Train single-model for each (facilities, model) combination and collect best
utility values. Outputs a Python dict ready to paste into analyse_metrics.py.
"""

import concurrent.futures
import os
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent

FACILITIES = [3, 5, 10]
MODELS = ["huff", "partially_binary", "pareto_huff"]
SAMPLES = 3

MAX_WORKERS = max(os.cpu_count() or 4, 6)

FACILITY_ITERATIONS = {3: 30000, 5: 100000, 10: 200000}


def run_one(fac, model, sample_idx):
    rank_file = PROJECT_ROOT / f"ranks_{model}_{fac}_{sample_idx}.dat"
    env = os.environ.copy()
    env["MAX_FACILITIES"] = str(fac)
    env["TRAINING_MODE"] = "true"
    env["BEHAVIOUR_MODEL"] = model
    env["RANK_FILE"] = str(rank_file)
    env["ITERATIONS"] = str(FACILITY_ITERATIONS[fac])

    try:
        proc = subprocess.run(
            ["go", "run", "."],
            capture_output=True,
            text=True,
            env=env,
            timeout=1800,
            cwd=str(PROJECT_ROOT),
        )
        if proc.returncode != 0:
            return {"fac": fac, "model": model, "sample": sample_idx,
                    "error": f"exit {proc.returncode}"}

        match = re.search(r"Best solution found: .*?\(([\d.]+)%\)", proc.stdout)
        if not match:
            return {"fac": fac, "model": model, "sample": sample_idx,
                    "error": "parse failed"}

        return {"fac": fac, "model": model, "sample": sample_idx,
                "value": float(match.group(1))}
    except subprocess.TimeoutExpired:
        return {"fac": fac, "model": model, "sample": sample_idx,
                "error": "timeout"}
    finally:
        if rank_file.exists():
            rank_file.unlink()


def main():
    tasks = []
    for fac in FACILITIES:
        for model in MODELS:
            for s in range(SAMPLES):
                tasks.append((fac, model, s))

    total = len(tasks)
    print(f"Running {total} training tasks ({len(FACILITIES)} sizes x {len(MODELS)} models x {SAMPLES} samples)")
    print(f"Max workers: {MAX_WORKERS}")
    for fac in FACILITIES:
        print(f"  {fac} facilities: {FACILITY_ITERATIONS[fac]} iterations")
    print()

    results_by_fac_model = {}
    for fac in FACILITIES:
        results_by_fac_model[fac] = {}
        for model in MODELS:
            results_by_fac_model[fac][model] = []

    completed = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {
            executor.submit(run_one, fac, model, s): (fac, model, s)
            for fac, model, s in tasks
        }

        for future in concurrent.futures.as_completed(future_map):
            completed += 1
            result = future.result()
            fac = result["fac"]
            model = result["model"]

            if "error" in result:
                print(f"[{completed}/{total}] {fac}/{model} sample {result['sample']}: ERROR ({result['error']})     ",
                      file=sys.stderr)
            else:
                results_by_fac_model[fac][model].append(result["value"])
                print(f"[{completed}/{total}] {fac}/{model} sample {result['sample']}: {result['value']:.6f}%")

    print("\nBEST_KNOWN = {")
    for fac in FACILITIES:
        values = []
        for model in MODELS:
            samples = results_by_fac_model[fac][model]
            best = max(samples) if samples else 0.0
            values.append(best)
        if all(v > 0 for v in values):
            vals_str = ", ".join(f"{v:.6f}" for v in values)
            print(f"    {fac}: [{vals_str}],")
        else:
            print(f"    # {fac}: missing data", file=sys.stderr)
    print("}")

    # Clean up any leftover rank files
    for fac in FACILITIES:
        for model in MODELS:
            for s in range(SAMPLES):
                p = PROJECT_ROOT / f"ranks_{model}_{fac}_{s}.dat"
                if p.exists():
                    p.unlink()


if __name__ == "__main__":
    main()