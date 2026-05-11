"""Visualise the history of interquartile means (IQM) of knee point objectives across iterations.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


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


def create_line_chart(iteration_to_objectives, output_dir, max_facilities, runs, iterations, log_period):
    iterations_list = sorted(iteration_to_objectives.keys())
    if not iterations_list:
        print('No checkpoint data to plot', file=sys.stderr)
        return

    fig, ax = plt.subplots(figsize=(14, 8))
    fig.suptitle(
        'IQM of knee point objectives over iterations',
        fontsize=22, y=1.02,
    )

    colours = ['#2166AC', '#B2182B', '#4DAF4A']

    overall_iqm_per_iter = []
    for it in iterations_list:
        all_objectives_at_iter = []
        for dim in range(3):
            vals = iteration_to_objectives[it][dim]
            all_objectives_at_iter.extend(vals)
        overall_iqm_per_iter.append(interquartile_mean(all_objectives_at_iter))

    for dim in range(3):
        iqm_vals = []
        for it in iterations_list:
            vals = iteration_to_objectives[it][dim]
            iqm_vals.append(interquartile_mean(vals))

        ax.plot(iterations_list, iqm_vals, color=colours[dim], linewidth=2.5,
                marker='o', markersize=5, markerfacecolor='white',
                markeredgewidth=1.5, markeredgecolor=colours[dim],
                label=MODEL_NAMES[dim], zorder=4)

    ax.plot(iterations_list, overall_iqm_per_iter, color='grey', linewidth=2,
            linestyle=(0, (8, 4)), zorder=5,
            label=f'Overall IQM')

    ax.set_xlabel('Iterations', fontsize=16)
    ax.set_ylabel('Market Share (%)', fontsize=16)
    ax.tick_params(axis='both', labelsize=15)
    ax.legend(fontsize=16, loc='lower right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    ax.set_xlim(0, iterations * 1.02)

    plt.tight_layout()
    outpath = output_dir / 'iqm_history_chart.png'
    plt.savefig(outpath, dpi=150, facecolor='white', bbox_inches='tight')
    plt.close()
    print(f'Saved {outpath}')

    print(f'\nIQM by iteration checkpoint ({runs} runs):')
    header = f"{'Iter':>8}"
    for name in MODEL_NAMES:
        header += f' {name:>12}'
    print(header)
    print('-' * 48)
    for it in iterations_list:
        row = f'{it:>8}'
        for dim in range(3):
            vals = iteration_to_objectives[it][dim]
            iqm = interquartile_mean(vals)
            row += f' {iqm:>12.4f}'
        print(row)


def main():
    script_dir = Path(__file__).parent
    parent_dir = script_dir.parent

    if len(sys.argv) >= 2:
        input_path = Path(sys.argv[1])
    else:
        input_path = parent_dir / 'iqm_checkpoints.json'

    if not input_path.exists():
        print(f'File not found: {input_path}', file=sys.stderr)
        print('Run the IQM experiment first: make run_iqm_experiment', file=sys.stderr)
        sys.exit(1)

    with open(input_path, 'r') as f:
        data = json.load(f)

    meta = data['meta']
    iteration_to_objectives = {
        int(k): {int(dim): vals for dim, vals in v.items()}
        for k, v in data['iteration_to_objectives'].items()
    }

    create_line_chart(
        iteration_to_objectives, parent_dir,
        meta['max_facilities'], meta['runs'],
        meta['iterations'], meta['log_period']
    )

    print('Intermediate IQM chart generated.')


if __name__ == '__main__':
    main()