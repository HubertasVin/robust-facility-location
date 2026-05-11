"""Run the robust-facility-location solver multiple times and collect knee points at regular intervals.
"""

import concurrent.futures
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path


RUNS = 100
MAX_FACILITIES = 10
ITERATIONS = 32000 
LOG_PERIOD = 800
MODEL_NAMES = ['Huff', 'PartiallyBinary', 'ParetoHuff']


PROJECT_ROOT = Path(__file__).parent.parent.parent


def run_one(run_index):
    env = os.environ.copy()
    env['MAX_FACILITIES'] = str(MAX_FACILITIES)
    env['ITERATIONS'] = str(ITERATIONS)
    env['TRAINING_MODE'] = 'false'
    env['JSON_MODE'] = 'true'
    env['LOG_PERIOD'] = str(LOG_PERIOD)
    env['POPULATION_SIZE'] = '10'

    try:
        proc = subprocess.run(
            ['go', 'run', '.'],
            capture_output=True, text=True, timeout=1800,
            env=env,
            cwd=str(PROJECT_ROOT),
        )
        snapshots = []
        for line in proc.stderr.split('\n'):
            line = line.strip()
            if line.startswith('{') and 'knee_point' in line:
                try:
                    snapshots.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        out = proc.stdout.strip()
        final_kp = None
        if out:
            try:
                final = json.loads(out)
                if 'knee_point' in final and final['knee_point']:
                    final_kp = final['knee_point']['objectives']
            except json.JSONDecodeError:
                pass

        return {
            'run': run_index,
            'snapshots': snapshots,
            'final_knee_point': final_kp,
            'error': None,
        }
    except subprocess.TimeoutExpired:
        return {'run': run_index, 'snapshots': [], 'final_knee_point': None,
                'error': 'timeout'}
    except Exception as e:
        return {'run': run_index, 'snapshots': [], 'final_knee_point': None,
                'error': str(e)}


def collect_checkpoints(all_results):
    iteration_to_objectives = defaultdict(lambda: defaultdict(list))

    for res in all_results:
        if res.get('error'):
            continue
        for snap in res.get('snapshots', []):
            it = snap.get('iteration')
            kp = snap.get('knee_point')
            if it is not None and kp and len(kp) == 3:
                for dim in range(3):
                    iteration_to_objectives[it][dim].append(kp[dim])

    return iteration_to_objectives


def main():
    output_dir = Path(__file__).parent.parent

    print(f'Running {RUNS} trials...')
    print(f'  Facilities: {MAX_FACILITIES}')
    print(f'  Iterations per run: {ITERATIONS}')
    print(f'  Snapshot period: {LOG_PERIOD}')
    print(f'  Expected checkpoints per run: ~{ITERATIONS // LOG_PERIOD}')
    print()

    max_workers = min(os.cpu_count() or 4, 10)
    results = []
    completed = 0
    errors = 0
    start_time = time.time()

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(run_one, i): i
            for i in range(RUNS)
        }

        for future in concurrent.futures.as_completed(future_map):
            run_idx = future_map[future]
            try:
                res = future.result()
                results.append(res)
                completed += 1
                if res.get('error'):
                    errors += 1
                if completed % 10 == 0 or completed == RUNS:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta = (RUNS - completed) / rate if rate > 0 else 0
                    snaps = len(res.get('snapshots', []))
                    print(f'[{completed}/{RUNS}] Run {run_idx}: {snaps} snapshots '
                          f'| elapsed={elapsed:.0f}s eta={eta:.0f}s '
                          f'| errors={errors}')
            except Exception as e:
                results.append({'run': run_idx, 'snapshots': [], 'error': str(e)})
                completed += 1
                errors += 1

    total_elapsed = time.time() - start_time
    total_snapshots = sum(len(r.get('snapshots', [])) for r in results)
    print(f'\nDone in {total_elapsed:.0f}s. {errors} errors, {total_snapshots} total snapshots.')

    iteration_to_objectives = collect_checkpoints(results)
    print(f'Checkpoints with data: {len(iteration_to_objectives)}')

    checkpoint_data = {
        'meta': {
            'max_facilities': MAX_FACILITIES,
            'runs': RUNS,
            'iterations': ITERATIONS,
            'log_period': LOG_PERIOD,
        },
        'iteration_to_objectives': {str(k): dict(v) for k, v in iteration_to_objectives.items()},
    }

    outpath = output_dir / 'iqm_checkpoints.json'
    with open(outpath, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)
    print(f'Saved checkpoints to {outpath}')


if __name__ == '__main__':
    main()