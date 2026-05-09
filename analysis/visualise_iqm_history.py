#!/usr/bin/env python3

import concurrent.futures
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


RUNS = 100
MAX_FACILITIES = 10
ITERATIONS = 32000
LOG_PERIOD = 800
MODEL_NAMES = ['Huff', 'PartiallyBinary', 'ParetoHuff']


def interquartile_mean(values):
    a = np.sort(np.array(values, dtype=float))
    n = len(a)
    if n < 4:
        return np.mean(a)
    q1 = int(np.floor(0.25 * n))
    q3 = int(np.ceil(0.75 * n))
    return np.mean(a[q1:q3])


def general_mean(values):
    return float(np.mean(np.array(values, dtype=float)))


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


def create_line_chart(iteration_to_objectives, output_dir):
    iterations = sorted(iteration_to_objectives.keys())
    if not iterations:
        print('No checkpoint data to plot', file=sys.stderr)
        return

    fig, ax = plt.subplots(figsize=(14, 8))
    fig.suptitle(
        'Interquartile Mean of Knee-Point Objectives over Iterations\n'
        f'({MAX_FACILITIES}-Facility Sets, {RUNS} Runs, Dashed = Overall Mean)',
        fontsize=15, y=1.02,
    )

    colours = ['#2166AC', '#B2182B', '#4DAF4A']

    overall_iqm_per_iter = []
    for it in iterations:
        all_objectives_at_iter = []
        for dim in range(3):
            vals = iteration_to_objectives[it][dim]
            all_objectives_at_iter.extend(vals)
        overall_iqm_per_iter.append(interquartile_mean(all_objectives_at_iter))

    for dim in range(3):
        iqm_vals = []
        for it in iterations:
            vals = iteration_to_objectives[it][dim]
            iqm_vals.append(interquartile_mean(vals))

        ax.plot(iterations, iqm_vals, color=colours[dim], linewidth=2.5,
                marker='o', markersize=5, markerfacecolor='white',
                markeredgewidth=1.5, markeredgecolor=colours[dim],
                label=MODEL_NAMES[dim], zorder=4)

    ax.plot(iterations, overall_iqm_per_iter, color='grey', linewidth=2,
            linestyle=(0, (8, 4)), zorder=5,
            label=f'Overall IQM')

    ax.set_xlabel('Iterations', fontsize=13)
    ax.set_ylabel('Market Share (%)', fontsize=13)
    ax.legend(fontsize=11, loc='lower right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    ax.set_xlim(0, ITERATIONS * 1.02)

    plt.tight_layout()
    outpath = Path(output_dir) / 'iqm_history_chart.png'
    plt.savefig(outpath, dpi=150, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f'Saved {outpath}')

    print(f'\nIQM by iteration checkpoint ({RUNS} runs):')
    header = f"{'Iter':>8}"
    for name in MODEL_NAMES:
        header += f' {name:>12}'
    print(header)
    print('-' * 48)
    for it in iterations:
        row = f'{it:>8}'
        for dim in range(3):
            vals = iteration_to_objectives[it][dim]
            iqm = interquartile_mean(vals)
            row += f' {iqm:>12.4f}'
        print(row)


def main():
    script_dir = Path(__file__).parent
    output_dir = script_dir

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
    create_line_chart(iteration_to_objectives, output_dir)

    print('Intermediate IQM chart generated.')


if __name__ == '__main__':
    main()