"""Analyze raw_results.json and compute metrics for each configuration.

Produces metrics_results.csv with columns:
max_facilities, iterations, solution_stability, hypervolume, price_of_robustness,
optimality_gap, coefficient_of_variation
"""

import csv
import json
import math
import sys
from collections import defaultdict
from itertools import combinations

BEST_KNOWN = {
    3: [21.800306, 29.292358, 23.080654],
    5: [27.748007, 35.009718, 29.088894],
    10: [35.011573, 40.522524, 36.247860],
}


def jaccard_similarity(set_a, set_b):
    a = set(set_a)
    b = set(set_b)
    if not a or not b:
        return 0.0
    intersection = a & b
    union = a | b
    if not union:
        return 0.0
    return len(intersection) / len(union)


def solution_stability(runs):
    location_sets = []
    for r in runs:
        if "knee_point" in r and r["knee_point"]:
            location_sets.append(set(r["knee_point"]["locations"]))
    n = len(location_sets)
    if n < 2:
        return 0.0
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += jaccard_similarity(location_sets[i], location_sets[j])
            count += 1
    return total / count if count > 0 else 0.0


def hypervolume(pareto_front, reference_point):
    pareto = [(s["objectives"][0], s["objectives"][1], s["objectives"][2])
              for s in pareto_front if "objectives" in s and len(s["objectives"]) == 3]
    if not pareto or not reference_point:
        return 0.0

    r1, r2, r3 = reference_point
    points = [(min(max(o1, 0), r1), min(max(o2, 0), r2), min(max(o3, 0), r3))
              for o1, o2, o3 in pareto]

    dominated = set()
    for o1, o2, o3 in points:
        for x in range(int(o1 * 100), int(r1 * 100) + 1):
            for y in range(int(o2 * 100), int(r2 * 100) + 1):
                for z in range(int(o3 * 100), int(r3 * 100) + 1):
                    dominated.add((x, y, z))

    total_volume = (r1 * 100) * (r2 * 100) * (r3 * 100)
    if total_volume == 0:
        return 0.0
    return len(dominated) / total_volume


def hypervolume_monte_carlo(pareto_front, reference_point, ideal_point, samples=20000):
    pareto = [(s["objectives"][0], s["objectives"][1], s["objectives"][2])
              for s in pareto_front if "objectives" in s and len(s["objectives"]) == 3]
    if not pareto or not reference_point or not ideal_point:
        return 0.0

    r1, r2, r3 = reference_point
    i1, i2, i3 = ideal_point
    import random
    random.seed(42)

    dominated_count = 0
    for _ in range(samples):
        x = random.uniform(r1, i1)
        y = random.uniform(r2, i2)
        z = random.uniform(r3, i3)
        for ox, oy, oz in pareto:
            if ox >= x and oy >= y and oz >= z:
                dominated_count += 1
                break

    volume_ratio = dominated_count / samples
    return volume_ratio


def mean(values):
    return sum(values) / len(values) if values else 0.0


def std_dev(values):
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def coefficient_of_variation(runs):
    objectives_per_run = []
    for r in runs:
        if "knee_point" in r and r["knee_point"] and "objectives" in r["knee_point"]:
            obj = r["knee_point"]["objectives"]
            if len(obj) == 3:
                objectives_per_run.append(obj)

    if len(objectives_per_run) < 2:
        return 0.0

    cvs = []
    for dim in range(3):
        values = [obj[dim] for obj in objectives_per_run]
        m = mean(values)
        s = std_dev(values)
        cvs.append(s / m if m != 0 else 0.0)

    return mean(cvs)


def price_of_robustness(runs, facilities):
    ratios = []
    best = BEST_KNOWN.get(facilities)
    if not best:
        return 0.0

    for r in runs:
        if "knee_point" in r and r["knee_point"] and "objectives" in r["knee_point"]:
            obj = r["knee_point"]["objectives"]
            if len(obj) == 3:
                f_robust = mean(obj)
                f_best = mean(best)
                if f_best != 0:
                    ratios.append(f_robust / f_best)

    return mean(ratios) if ratios else 0.0


def optimality_gap(runs, facilities):
    gaps = []
    best = BEST_KNOWN.get(facilities)
    if not best:
        return 0.0

    for r in runs:
        if "knee_point" in r and r["knee_point"] and "objectives" in r["knee_point"]:
            obj = r["knee_point"]["objectives"]
            if len(obj) == 3:
                for dim in range(3):
                    f_robust = obj[dim]
                    f_best = best[dim]
                    if f_robust != 0:
                        gaps.append((f_robust - f_best) / f_robust * 100)

    return mean(gaps) if gaps else 0.0


def reference_point_from_runs(runs):
    min_vals = [float("inf")] * 3
    for r in runs:
        if "pareto_front" in r and r["pareto_front"]:
            for sol in r["pareto_front"]:
                if "objectives" in sol and len(sol["objectives"]) == 3:
                    for dim in range(3):
                        if sol["objectives"][dim] < min_vals[dim]:
                            min_vals[dim] = sol["objectives"][dim]
    if min_vals[0] == float("inf"):
        return None
    return [v * 0.9 for v in min_vals]


def ideal_point_from_runs(runs):
    max_vals = [float("-inf")] * 3
    for r in runs:
        if "pareto_front" in r and r["pareto_front"]:
            for sol in r["pareto_front"]:
                if "objectives" in sol and len(sol["objectives"]) == 3:
                    for dim in range(3):
                        if sol["objectives"][dim] > max_vals[dim]:
                            max_vals[dim] = sol["objectives"][dim]
    if max_vals[0] == float("-inf"):
        return None
    return max_vals


def main():
    if len(sys.argv) < 2:
        input_file = "raw_results.json"
    else:
        input_file = sys.argv[1]

    if len(sys.argv) >= 3:
        output_file = sys.argv[2]
    else:
        output_file = "metrics_results.csv"

    with open(input_file, "r") as f:
        all_results = json.load(f)

    groups = defaultdict(list)
    for r in all_results:
        key = (r.get("max_facilities", 0),
               r.get("iterations", 0))
        groups[key].append(r)

    rows = []
    for (fac, it), runs in sorted(groups.items()):
        valid_runs = [r for r in runs if "error" not in r]
        n_valid = len(valid_runs)

        ref_point = reference_point_from_runs(valid_runs)
        ideal_point = ideal_point_from_runs(valid_runs)

        hv_values = []
        for r in valid_runs:
            if "pareto_front" in r and r["pareto_front"] and ref_point and ideal_point:
                hv = hypervolume_monte_carlo(r["pareto_front"], ref_point, ideal_point)
                hv_values.append(hv)
        hv = mean(hv_values) if hv_values else 0.0

        rows.append({
            "max_facilities": fac,
            "iterations": it,
            "sample_count": n_valid,
            "solution_stability": round(solution_stability(valid_runs), 6),
            "hypervolume": round(hv, 6),
            "price_of_robustness": round(price_of_robustness(valid_runs, fac), 6),
            "optimality_gap": round(optimality_gap(valid_runs, fac), 6),
            "coefficient_of_variation": round(coefficient_of_variation(valid_runs), 6),
        })

    fieldnames = [
        "max_facilities",
        "iterations",
        "sample_count",
        "solution_stability",
        "hypervolume",
        "price_of_robustness",
        "optimality_gap",
        "coefficient_of_variation",
    ]

    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_file}")

    print("\nPreview:")
    print(f"{'Facilities':>10} {'Iter':>8} {'Stability':>10} {'HV':>8} {'PoR':>8} {'GAP':>8} {'CV':>8}")
    print("-" * 68)
    for row in rows:
        print(f"{row['max_facilities']:>10} {row['iterations']:>8} "
              f"{row['solution_stability']:>10.4f} {row['hypervolume']:>8.4f} "
              f"{row['price_of_robustness']:>8.4f} {row['optimality_gap']:>8.4f} "
              f"{row['coefficient_of_variation']:>8.4f}")


if __name__ == "__main__":
    main()
