"""Run the robust-facility-location solver with multiple configurations in parallel.

Collects 20 samples per (iterations, max_facilities) combination and saves
raw JSON results to raw_results.json.
"""

import concurrent.futures
import json
import os
import subprocess
import sys
import time
from itertools import product

ITERATIONS_LIST = [3000, 15000, 50000]
FACILITIES_LIST = [3, 5, 10]
SAMPLES = 20
MAX_WORKERS = min(os.cpu_count() or 4, 10)

def run_one(iterations, facilities, run_index):
    env = os.environ.copy()
    env["ITERATIONS"] = str(iterations)
    env["MAX_FACILITIES"] = str(facilities)
    env["TRAINING_MODE"] = "false"
    env["JSON_MODE"] = "true"

    stderr_lines = []
    try:
        proc = subprocess.run(
            ["go", "run", "."],
            capture_output=True,
            text=True,
            env=env,
            timeout=1800,
        )
        if proc.returncode != 0:
            return {
                "iterations": iterations,
                "max_facilities": facilities,
                "run": run_index,
                "error": f"exit code {proc.returncode}",
                "stderr": proc.stderr[-2000:],
            }
        return json.loads(proc.stdout)
    except subprocess.TimeoutExpired:
        return {
            "iterations": iterations,
            "max_facilities": facilities,
            "run": run_index,
            "error": "timeout after 1800s",
        }
    except json.JSONDecodeError as e:
        return {
            "iterations": iterations,
            "max_facilities": facilities,
            "run": run_index,
            "error": f"JSON decode error: {e}",
        }


def main():
    combinations = list(product(ITERATIONS_LIST, FACILITIES_LIST))
    total_runs = len(combinations) * SAMPLES

    print(f"Running {total_runs} experiments ({SAMPLES} samples × {len(combinations)} configurations)")
    print(f"Max parallel workers: {MAX_WORKERS}")
    print()

    tasks = []
    for iterations, facilities in combinations:
        for i in range(SAMPLES):
            tasks.append((iterations, facilities, i))

    results = []
    completed = 0
    start_time = time.time()

    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_map = {
            executor.submit(run_one, it, fac, idx): (it, fac, idx)
            for it, fac, idx in tasks
        }

        for future in concurrent.futures.as_completed(future_map):
            it, fac, idx = future_map[future]
            try:
                result = future.result()
                results.append(result)
                completed += 1
                if completed % 10 == 0 or completed == total_runs:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta = (total_runs - completed) / rate if rate > 0 else 0
                    print(f"Completed: {completed}/{total_runs} "
                          f"({completed / total_runs * 100:.1f}%) "
                          f"Elapsed: {elapsed:.0f}s ETA: {eta:.0f}s Rate: {rate:.2f} runs/s")
            except Exception as e:
                results.append({
                    "iterations": it,
                    "max_facilities": fac,
                    "run": idx,
                    "error": str(e),
                })
                completed += 1

    output_file = "raw_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    total_elapsed = time.time() - start_time
    errors = sum(1 for r in results if "error" in r)
    print(f"\nDone! {total_elapsed:.1f}s total. "
          f"Results: {len(results)} runs, {errors} errors -> {output_file}")


if __name__ == "__main__":
    main()