#!/usr/bin/env python3

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
import csv
from dataclasses import dataclass
from typing import List, Tuple, Optional
from pathlib import Path


@dataclass
class Solution:
    locations: str
    huff_model: float
    partially_binary: float

    @property
    def objectives(self) -> np.ndarray:
        return np.array([self.huff_model, self.partially_binary])


def load_solutions(filepath: str) -> List[Solution]:
    solutions = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            solutions.append(Solution(
                locations=row['locations'],
                huff_model=float(row['HuffModel']),
                partially_binary=float(row['PartiallyBinaryModel'])
            ))
    return solutions


def normalise_objectives(solutions: List[Solution]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    objectives = np.array([s.objectives for s in solutions])
    mins = objectives.min(axis=0)
    maxs = objectives.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1
    return (objectives - mins) / ranges, mins, maxs


def find_pareto_front(objectives: np.ndarray) -> np.ndarray:
    n = len(objectives)
    is_dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i != j and np.all(objectives[j] >= objectives[i]) and np.any(objectives[j] > objectives[i]):
                is_dominated[i] = True
                break
    pareto = np.where(~is_dominated)[0]
    if len(pareto) < 4:
        sorted_by_huff = np.argsort(objectives[:, 0])[::-1]
        pareto = np.unique(np.concatenate([pareto, sorted_by_huff[:4]]))
    return np.sort(pareto)


def find_knee_point(objectives: np.ndarray) -> int:
    ideal = np.array([1.0, 1.0])
    return np.argmin(np.linalg.norm(objectives - ideal, axis=1))


def visualise(filepath: str, output_path: Optional[str] = None):
    solutions = load_solutions(filepath)
    if len(solutions) == 0:
        print("No solutions found.")
        return

    normalised, _, _ = normalise_objectives(solutions)

    pareto_indices = find_pareto_front(normalised)
    pareto_objectives = normalised[pareto_indices]
    dominated_indices = list(set(range(len(normalised))) - set(pareto_indices))

    knee_idx = find_knee_point(pareto_objectives)
    knee_global_idx = pareto_indices[knee_idx]
    knee_coords = normalised[knee_global_idx]
    ideal_coords = np.array([1.0, 1.0])

    if len(dominated_indices) > 4:
        distances = [(idx, np.min(np.linalg.norm(pareto_objectives - normalised[idx], axis=1))) for idx in dominated_indices]
        distances.sort(key=lambda x: x[1])
        selected_dominated = [idx for idx, _ in distances[:4]]
    else:
        selected_dominated = dominated_indices[:4]

    other_mask = np.any(pareto_objectives != knee_coords, axis=1) & np.any(pareto_objectives != ideal_coords, axis=1)
    other_pareto_indices = np.where(other_mask)[0]
    if len(other_pareto_indices) > 4:
        other_pareto_indices = other_pareto_indices[::2]

    fig, ax = plt.subplots(figsize=(12, 10))
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.98)

    all_points = np.array(list(pareto_objectives) + [normalised[idx] for idx in selected_dominated] + [np.array([1.0, 1.0])])
    data_min, data_max = all_points.min(axis=0), all_points.max(axis=0)

    ax.set_xlim(data_min[0] - 0.01, data_max[0] + 0.01)
    ax.set_ylim(data_min[1] - 0.01, data_max[1] + 0.01)

    if selected_dominated:
        for idx in selected_dominated:
            coord = normalised[idx]
            ax.scatter(coord[0], coord[1], c='lightgray', s=225,
                       alpha=0.9, edgecolors='black', linewidth=1.5)
        ax.scatter([], [], c='lightgray', s=225, edgecolors='black',
                  linewidth=1.5, label='Dominated solutions')

    if len(other_pareto_indices) > 0:
        for idx in other_pareto_indices:
            coord = pareto_objectives[idx]
            ax.scatter(coord[0], coord[1], c='royalblue', s=225,
                       alpha=0.9, edgecolors='black', linewidth=1.5)
        ax.scatter([], [], c='royalblue', s=225, edgecolors='black',
                  linewidth=1.5, label='Pareto front')

    if len(pareto_objectives) > 0:
        sorted_pareto = pareto_objectives[np.argsort(pareto_objectives[:, 0])]
        ax.plot(sorted_pareto[:, 0], sorted_pareto[:, 1], 'royalblue',
                linewidth=2, alpha=0.8)

        ax.plot([knee_coords[0], 1.0], [knee_coords[1], 1.0], 'k--',
                linewidth=2, alpha=0.7)

    ax.scatter(knee_coords[0], knee_coords[1], c='gold', s=600,
               marker='*', edgecolors='black', linewidth=2)

    ax.scatter(1.0, 1.0, c='lime', s=375, marker='D',
               edgecolors='black', linewidth=2)

    ax.scatter([], [], c='gold', s=200, marker='*',
              edgecolors='black', linewidth=2, label='Knee point')
    ax.scatter([], [], c='lime', s=150, marker='D',
              edgecolors='black', linewidth=2, label='Ideal point')

    ax.set_xlabel('Huff Model', fontsize=13, labelpad=12)
    ax.set_ylabel('Partially Binary Model', fontsize=13, labelpad=12)
    ax.set_title('2D Pareto Front: Huff vs PartiallyBinary', fontsize=16, pad=25)

    legend = ax.legend(loc='upper left', fontsize=10, framealpha=0.95,
                       labelspacing=0.75, handletextpad=0.5, borderpad=0.75)
    legend.get_frame().set_linewidth(1.5)

    knee_sol = solutions[knee_global_idx]
    info_text = (f'Knee point: {{{knee_sol.locations}}}\n'
                 f'Huff: {knee_sol.huff_model:.2f}%  |  P.Binary: {knee_sol.partially_binary:.2f}%')

    ax.text(1.0, 0.98, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95,
                      edgecolor='gold', linewidth=2))

    if output_path:
        plt.savefig(output_path, dpi=150, facecolor='white')
        print(f"Figure saved to {output_path}")

    plt.close()


if __name__ == '__main__':
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent

    filepath = sys.argv[1] if len(sys.argv) > 1 else str(project_root / 'checked_solutions.tsv')
    output_path = sys.argv[2] if len(sys.argv) > 2 else str(script_dir.parent / 'pareto_2d.png')
    print(f"Loading solutions from {filepath}...")
    visualise(filepath, output_path)