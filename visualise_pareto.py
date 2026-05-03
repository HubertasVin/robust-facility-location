#!/usr/bin/env python3
"""Visualizer for Pareto front and knee point identification."""

import sys
import numpy as np
import matplotlib

def _set_backend():
    """Set matplotlib backend with proper fallback."""
    backends_to_try = ['Qt5Agg', 'TkAgg', 'Agg']
    for backend in backends_to_try:
        try:
            matplotlib.use(backend, force=True)
            # Test if the backend can actually be imported
            import importlib
            if backend == 'Qt5Agg':
                # Try to import Qt binding
                try:
                    importlib.import_module('PyQt5')
                except ImportError:
                    try:
                        importlib.import_module('PySide2')
                    except ImportError:
                        raise ImportError("No Qt binding available")
            elif backend == 'TkAgg':
                importlib.import_module('tkinter')
            # If we get here, the backend should work
            return
        except (ImportError, ModuleNotFoundError):
            continue
    # Fallback to Agg if nothing else works
    matplotlib.use('Agg', force=True)

_set_backend()
import matplotlib.pyplot as plt
import csv
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class Solution:
    locations: str
    huff_model: float
    partially_binary: float
    pareto_huff: float
    
    @property
    def objectives(self) -> np.ndarray:
        return np.array([self.huff_model, self.partially_binary, self.pareto_huff])


def load_solutions(filepath: str) -> List[Solution]:
    solutions = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            solutions.append(Solution(
                locations=row['locations'],
                huff_model=float(row['HuffModel']),
                partially_binary=float(row['PartiallyBinaryModel']),
                pareto_huff=float(row['ParetoHuffModel'])
            ))
    return solutions


def normalize_objectives(solutions: List[Solution]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    return np.where(~is_dominated)[0]


def find_extreme_solutions(objectives: np.ndarray) -> List[int]:
    return [np.argmax(objectives[:, k]) for k in range(objectives.shape[1])]


def compute_hyperplane(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> Tuple[np.ndarray, float]:
    v1, v2 = p1 - p0, p2 - p0
    normal = np.cross(v1, v2)
    ideal = np.array([1.0, 1.0, 1.0])
    if np.dot(normal, ideal - p0) < 0:
        normal = -normal
    return normal, np.dot(normal, p0)


def find_knee_point(objectives: np.ndarray, extremes: List[int]) -> int:
    if len(set(extremes)) < 3:
        ideal = np.array([1.0, 1.0, 1.0])
        return np.argmin(np.linalg.norm(objectives - ideal, axis=1))
    
    normal, d = compute_hyperplane(objectives[extremes[0]], objectives[extremes[1]], objectives[extremes[2]])
    normal_norm = np.linalg.norm(normal)
    distances = [np.dot(normal, obj) - d for obj in objectives]
    return np.argmax(distances)


def create_hyperplane_vertices(extremes_coords: np.ndarray, bounds_min: np.ndarray, bounds_max: np.ndarray) -> List[np.ndarray]:
    p0, p1, p2 = extremes_coords[0], extremes_coords[1], extremes_coords[2]
    normal = np.cross(p1 - p0, p2 - p0)
    normal_norm = np.linalg.norm(normal)
    if normal_norm == 0:
        return []
    normal = normal / normal_norm
    ideal = np.array([1.0, 1.0, 1.0])
    if np.dot(normal, ideal - p0) < 0:
        normal = -normal
    d = np.dot(normal, p0)
    
    vertices = []
    for y in [bounds_min[1], bounds_max[1]]:
        for z in [bounds_min[2], bounds_max[2]]:
            if abs(normal[0]) > 1e-10:
                x = (d - normal[1]*y - normal[2]*z) / normal[0]
                if bounds_min[0] <= x <= bounds_max[0]:
                    vertices.append(np.array([x, y, z]))
    for x in [bounds_min[0], bounds_max[0]]:
        for z in [bounds_min[2], bounds_max[2]]:
            if abs(normal[1]) > 1e-10:
                y = (d - normal[0]*x - normal[2]*z) / normal[1]
                if bounds_min[1] <= y <= bounds_max[1]:
                    vertices.append(np.array([x, y, z]))
    for x in [bounds_min[0], bounds_max[0]]:
        for y in [bounds_min[1], bounds_max[1]]:
            if abs(normal[2]) > 1e-10:
                z = (d - normal[0]*x - normal[1]*y) / normal[2]
                if bounds_min[2] <= z <= bounds_max[2]:
                    vertices.append(np.array([x, y, z]))
    
    unique = []
    for v in vertices:
        if not any(np.linalg.norm(v - uv) < 1e-6 for uv in unique):
            unique.append(v)
    return unique


def order_vertices_ccw(vertices: List[np.ndarray], normal: np.ndarray) -> List[np.ndarray]:
    if len(vertices) < 3:
        return vertices
    centroid = np.mean(vertices, axis=0)
    v1 = (vertices[0] - centroid) / np.linalg.norm(vertices[0] - centroid)
    v2 = np.cross(normal, v1)
    angles = [np.arctan2(np.dot(v - centroid, v2), np.dot(v - centroid, v1)) for v in vertices]
    return [vertices[i] for i in np.argsort(angles)]


def blend_with_gray(hex_color: str, blend_factor: float = 0.5) -> str:
    hex_color = hex_color.lstrip('#')
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    gray = 128
    return f'#{int(r * (1 - blend_factor) + gray * blend_factor):02x}{int(g * (1 - blend_factor) + gray * blend_factor):02x}{int(b * (1 - blend_factor) + gray * blend_factor):02x}'


def visualize(filepath: str, output_path: Optional[str] = None):
    solutions = load_solutions(filepath)
    normalized, _, _ = normalize_objectives(solutions)
    
    pareto_indices = find_pareto_front(normalized)
    pareto_objectives = normalized[pareto_indices]
    dominated_indices = list(set(range(len(normalized))) - set(pareto_indices))
    
    extreme_indices = find_extreme_solutions(pareto_objectives)
    extreme_global_indices = pareto_indices[extreme_indices]
    extreme_coords = normalized[extreme_global_indices]
    
    knee_idx = find_knee_point(pareto_objectives, extreme_indices)
    knee_global_idx = pareto_indices[knee_idx]
    knee_coords = normalized[knee_global_idx]
    
    if len(dominated_indices) > 4:
        distances = [(idx, np.min(np.linalg.norm(pareto_objectives - normalized[idx], axis=1))) for idx in dominated_indices]
        distances.sort(key=lambda x: x[1])
        selected_dominated = [idx for idx, _ in distances[:4]]
    else:
        selected_dominated = dominated_indices[:4]
    
    other_mask = np.ones(len(pareto_indices), dtype=bool)
    other_mask[knee_idx] = False
    other_mask[extreme_indices] = False
    other_pareto_indices = np.where(other_mask)[0][::2]
    
    fig = plt.figure(figsize=(14, 11))
    ax = fig.add_subplot(111, projection='3d')
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.05, right=0.98)
    
    normal, d = compute_hyperplane(extreme_coords[0], extreme_coords[1], extreme_coords[2])
    normal_norm = np.linalg.norm(normal)
    if normal_norm == 0:
        # Degenerate case: normal is zero, use default unit vector
        normal_unit = np.array([1.0, 0.0, 0.0])
    else:
        normal_unit = normal / normal_norm
    
    relevant_coords = np.array(list(pareto_objectives) + [normalized[idx] for idx in selected_dominated] + [np.array([1.0, 1.0, 1.0])])
    data_min, data_max = relevant_coords.min(axis=0), relevant_coords.max(axis=0)
    padding = (data_max - data_min) * 0.15
    min_coords, max_coords = data_min - padding, data_max + padding
    
    ax.set_xlim(min_coords[0], max_coords[0])
    ax.set_ylim(min_coords[1], max_coords[1])
    ax.set_zlim(min_coords[2], max_coords[2])
    
    hyperplane_vertices = create_hyperplane_vertices(extreme_coords, min_coords, max_coords)
    if len(hyperplane_vertices) >= 3:
        ordered = order_vertices_ccw(hyperplane_vertices, normal_unit)
        if len(ordered) >= 3:
            verts = np.array(ordered)
            closed = np.vstack([verts, verts[0]])
            ax.plot(closed[:, 0], closed[:, 1], closed[:, 2], 'darkblue', linewidth=2)
            
            centroid = np.mean(ordered, axis=0)
            plane_normal = normal_unit
            basis1 = (np.array(ordered[1]) - np.array(ordered[0]))
            basis1 = basis1 / np.linalg.norm(basis1)
            basis2 = np.cross(plane_normal, basis1)
            basis2 = basis2 / np.linalg.norm(basis2)
            
            proj1 = [np.dot(v - centroid, basis1) for v in verts]
            proj2 = [np.dot(v - centroid, basis2) for v in verts]
            min1, max1, min2, max2 = min(proj1), max(proj1), min(proj2), max(proj2)
            
            for i in range(25):
                offset2 = min2 + (max2 - min2) * i / 24
                start = centroid + basis2 * offset2 + basis1 * (min1 - 0.05)
                end = centroid + basis2 * offset2 + basis1 * (max1 + 0.05)
                
                stripe_points = []
                n_verts = len(verts)
                for j in range(n_verts):
                    v1, v2 = verts[j], verts[(j + 1) % n_verts]
                    d1, d2 = end - start, v2 - v1
                    denom = d1[0] * d2[1] - d1[1] * d2[0]
                    if abs(denom) > 1e-10:
                        t = ((v1[0] - start[0]) * d2[1] - (v1[1] - start[1]) * d2[0]) / denom
                        s = ((v1[0] - start[0]) * d1[1] - (v1[1] - start[1]) * d1[0]) / denom
                        if 0 <= t <= 1 and 0 <= s <= 1:
                            stripe_points.append((t, start + t * d1))
                
                if len(stripe_points) >= 2:
                    stripe_points.sort(key=lambda x: x[0])
                    p1, p2 = stripe_points[0][1], stripe_points[-1][1]
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'darkblue', linewidth=2, alpha=0.6)
    
    behind_dominated = [np.dot(normal, normalized[idx]) < d for idx in selected_dominated]
    behind_pareto = [np.dot(normal, pareto_objectives[idx]) < d for idx in other_pareto_indices]
    border_behind = 'dimgray'
    
    sizes = {'dominated': 225, 'pareto': 225, 'extreme': 300, 'knee': 600, 'ideal': 375}
    legend_sizes = {'extreme': 150, 'knee': 200, 'ideal': 150}
    
    if selected_dominated:
        for i, idx in enumerate(selected_dominated):
            coord = normalized[idx]
            color = blend_with_gray('#D3D3D3', 0.5) if behind_dominated[i] else 'lightgray'
            edge = border_behind if behind_dominated[i] else 'black'
            ax.scatter([coord[0]], [coord[1]], [coord[2]], c=color, s=sizes['dominated'], 
                      alpha=0.9, edgecolors=edge, linewidth=1.5, depthshade=False)
        ax.scatter([], [], [], c='lightgray', s=sizes['dominated'], edgecolors='black', 
                  linewidth=1.5, label='Dominated solutions', depthshade=False)
    
    if len(other_pareto_indices) > 0:
        for i, idx in enumerate(other_pareto_indices):
            coord = pareto_objectives[idx]
            color = blend_with_gray('#4169E1', 0.5) if behind_pareto[i] else 'royalblue'
            edge = border_behind if behind_pareto[i] else 'black'
            ax.scatter([coord[0]], [coord[1]], [coord[2]], c=color, s=sizes['pareto'],
                      alpha=0.9, edgecolors=edge, linewidth=1.5, depthshade=False)
        ax.scatter([], [], [], c='royalblue', s=sizes['pareto'], edgecolors='black',
                  linewidth=1.5, label='Pareto front', depthshade=False)
    
    for coord in extreme_coords:
        ax.scatter([coord[0]], [coord[1]], [coord[2]], c='green', s=sizes['extreme'],
                  marker='s', edgecolors='black', linewidth=2, depthshade=False)
    ax.scatter([], [], [], c='green', s=legend_sizes['extreme'], marker='s',
              edgecolors='black', linewidth=2, label='Extreme solutions', depthshade=False)
    
    ax.scatter([knee_coords[0]], [knee_coords[1]], [knee_coords[2]], c='gold', s=sizes['knee'],
              marker='*', edgecolors='black', linewidth=2, depthshade=False)
    ax.scatter([], [], [], c='gold', s=legend_sizes['knee'], marker='*',
              edgecolors='black', linewidth=2, label='Knee point', depthshade=False)
    
    ax.scatter([1.0], [1.0], [1.0], c='lime', s=sizes['ideal'], marker='D',
              edgecolors='black', linewidth=2, depthshade=False)
    ax.scatter([], [], [], c='lime', s=legend_sizes['ideal'], marker='D',
              edgecolors='black', linewidth=2, label='Ideal point', depthshade=False)
    
    ax.plot([knee_coords[0], 1.0], [knee_coords[1], 1.0], [knee_coords[2], 1.0], 'k--', linewidth=2, alpha=0.7)
    for i in range(3):
        j = (i + 1) % 3
        ax.plot([extreme_coords[i, 0], extreme_coords[j, 0]],
               [extreme_coords[i, 1], extreme_coords[j, 1]],
               [extreme_coords[i, 2], extreme_coords[j, 2]], 'g-', linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Huff Model', fontsize=13, labelpad=12)
    ax.set_ylabel('Partially Binary Model', fontsize=13, labelpad=12)
    ax.set_zlabel('Pareto-Huff Model', fontsize=13, labelpad=12)
    ax.set_title('Knee Point Identification for Robust Facility Location', fontsize=16, pad=25)
    
    legend = ax.legend(loc='upper left', fontsize=10, framealpha=0.95,
                      bbox_to_anchor=(0.0, 0.98), labelspacing=0.75, handletextpad=0.5, borderpad=0.75)
    legend.get_frame().set_linewidth(1.5)
    
    extreme_locations = [solutions[pareto_indices[i]].locations for i in extreme_indices]
    knee_sol = solutions[knee_global_idx]
    extreme_lines = '\n'.join([f'  {{{loc}}}' for loc in extreme_locations])
    info_text = (f'Knee point: {{{knee_sol.locations}}}\n'
                f'Huff: {knee_sol.huff_model:.2f}%  |  P.Binary: {knee_sol.partially_binary:.2f}%  |  P.Huff: {knee_sol.pareto_huff:.2f}%\n'
                f'Extreme solutions:\n{extreme_lines}')
    
    ax.text2D(1.0, 0.98, info_text, transform=ax.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95, edgecolor='gold', linewidth=2),
             clip_on=False)
    
    azim = np.degrees(np.arctan2(normal_unit[1], normal_unit[0]))
    elev = np.degrees(np.arcsin(np.clip(normal_unit[2], -1, 1)))
    ax.view_init(elev=20 + elev * 0.3, azim=azim)
    
    if output_path:
        plt.savefig(output_path, dpi=150, facecolor='white')
        print(f"Initial figure saved to {output_path}")
    
    print("\nInteractive mode: Rotate and zoom the 3D view with your mouse.")
    print("Close the window when done.")
    plt.show()


if __name__ == '__main__':
    filepath = sys.argv[1] if len(sys.argv) > 1 else 'checked_solutions.tsv'
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    print(f"Loading solutions from {filepath}...")
    visualize(filepath, output_path)