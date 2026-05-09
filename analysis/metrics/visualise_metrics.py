#!/usr/bin/env python3

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import csv
from collections import defaultdict
from pathlib import Path


METRIC_LABELS = {
    'solution_stability': 'Solution Stability',
    'hypervolume': 'Hypervolume',
    'optimality_gap': 'Optimality Gap (%)',
}


def load_metrics(filepath):
    rows = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def extract_groups(rows):
    facility_sizes = sorted(set(int(r['max_facilities']) for r in rows))
    iteration_counts = sorted(set(int(r['iterations']) for r in rows))
    return facility_sizes, iteration_counts


def build_data(rows, facility_sizes, iteration_counts, metric):
    data = defaultdict(dict)
    for fsize in facility_sizes:
        for icount in iteration_counts:
            data[fsize][icount] = None
    for row in rows:
        fsize = int(row['max_facilities'])
        icount = int(row['iterations'])
        data[fsize][icount] = float(row[metric])
    return data


def format_iter_label(n):
    if n >= 1000:
        return f'{n // 1000}k'
    return str(n)


def create_chart(data, facility_sizes, iteration_counts, metric, output_dir):
    label = METRIC_LABELS.get(metric, metric)
    filename = f'{metric}_chart.png'

    x = np.arange(len(facility_sizes))
    bar_width = 0.8 / len(iteration_counts)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(iteration_counts)))

    fig, ax = plt.subplots(figsize=(12, 7))

    for i, icount in enumerate(iteration_counts):
        values = [data[fsize].get(icount, 0) for fsize in facility_sizes]
        offset = (i - (len(iteration_counts) - 1) / 2) * bar_width
        bars = ax.bar(
            x + offset,
            values,
            bar_width,
            label=format_iter_label(icount),
            color=colors[i],
            edgecolor='black',
            linewidth=0.8,
        )
        for bar, val in zip(bars, values):
            if val is not None:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                    f'{val:.3f}',
                    ha='center',
                    va='bottom',
                    fontsize=14,
                )

    tick_fontsize = 14
    label_fontsize = 15

    ax.set_xlabel('Facility Set Size', fontsize=label_fontsize)
    ax.set_ylabel(label, fontsize=label_fontsize)
    ax.set_title(f'{label} by Facility Set Size and Iterations', fontsize=15, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([str(fs) for fs in facility_sizes], fontsize=tick_fontsize)
    ax.tick_params(axis='y', labelsize=tick_fontsize)
    ax.legend(title='Iterations', fontsize=13, title_fontsize=14, loc='upper left')

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 1.15)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    outpath = output_dir / filename
    plt.savefig(outpath, dpi=150, facecolor='white')
    plt.close()
    print(f'Saved {outpath}')


def main():
    script_dir = Path(__file__).parent
    parent_dir = script_dir.parent
    csv_path = parent_dir / 'robustness_metrics.csv'

    rows = load_metrics(csv_path)
    facility_sizes, iteration_counts = extract_groups(rows)

    metrics = [
        'solution_stability',
        'hypervolume',
        'optimality_gap',
    ]

    for metric in metrics:
        data = build_data(rows, facility_sizes, iteration_counts, metric)
        create_chart(data, facility_sizes, iteration_counts, metric, parent_dir)

    print('All charts generated.')


if __name__ == '__main__':
    main()