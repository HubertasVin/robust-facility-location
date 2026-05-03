package solution

import (
	"math"

	"gonum.org/v1/gonum/spatial/r3"
)

// ParetoFront represents a collection of non-dominated solutions
type ParetoFront struct {
	Solutions []*Solution
}

// NewParetoFront creates an empty Pareto front
func NewParetoFront() *ParetoFront {
	return &ParetoFront{
		Solutions: make([]*Solution, 0),
	}
}

// AddSolution attempts to add a solution to the Pareto front
// Returns true if the solution was added (is non-dominated)
func (pf *ParetoFront) AddSolution(solution *Solution) bool {
	// Check if solution is dominated by any solution in the front
	for _, existing := range pf.Solutions {
		if existing.Dominates(solution) {
			return false
		}
		// Skip if solution has identical objectives to an existing one
		if objectivesEqual(existing.Objectives, solution.Objectives) {
			return false
		}
	}

	// Remove any solutions that are dominated by the new solution
	var nonDominated []*Solution
	for _, existing := range pf.Solutions {
		if !solution.Dominates(existing) {
			nonDominated = append(nonDominated, existing)
		}
	}

	// Add the new solution
	nonDominated = append(nonDominated, solution)
	pf.Solutions = nonDominated
	return true
}

// objectivesEqual checks if two objective slices are equal element-wise
func objectivesEqual(a, b []float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// CalculateIdealPoint calculates the ideal point from a set of solutions
// The ideal point has the maximum value for each objective
func CalculateIdealPoint(solutions []*Solution) []float64 {
	if len(solutions) == 0 {
		return nil
	}

	numObjectives := len(solutions[0].Objectives)
	idealPoint := make([]float64, numObjectives)

	// Initialize with minimum values
	for i := range idealPoint {
		idealPoint[i] = math.Inf(-1)
	}

	// Find maximum for each objective
	for _, solution := range solutions {
		for i, obj := range solution.Objectives {
			if obj > idealPoint[i] {
				idealPoint[i] = obj
			}
		}
	}

	return idealPoint
}

func (pf *ParetoFront) findClosestToIdealPoint() *Solution {
	idealPoint := CalculateIdealPoint(pf.Solutions)

	// Find solution with minimum distance to ideal point
	kneePoint := pf.Solutions[0]
	minDistance := kneePoint.CalculateDistanceToIdeal(idealPoint)

	for _, solution := range pf.Solutions[1:] {
		distance := solution.CalculateDistanceToIdeal(idealPoint)
		if distance < minDistance {
			minDistance = distance
			kneePoint = solution
		}
	}

	return kneePoint.Copy()
}

func (pf *ParetoFront) findKneePointByHyperplane() *Solution {
	n := len(pf.Solutions)
	if n == 0 {
		return nil
	}

	m := len(pf.Solutions[0].Objectives)
	if m != 3 {
		return nil
	}

	mins := make([]float64, m)
	maxs := make([]float64, m)
	for k := range m {
		mins[k] = math.Inf(1)
		maxs[k] = math.Inf(-1)
	}

	for _, s := range pf.Solutions {
		if len(s.Objectives) != m {
			return nil
		}
		for k, obj := range s.Objectives {
			if obj < mins[k] {
				mins[k] = obj
			}
			if obj > maxs[k] {
				maxs[k] = obj
			}
		}
	}

	norm := func(obj float64, k int) float64 {
		r := maxs[k] - mins[k]
		if r <= 0 {
			return 0
		}
		v := (obj - mins[k]) / r
		if v < 0 {
			return 0
		}
		if v > 1 {
			return 1
		}
		return v
	}

	// Pick extreme solutions: one that maximizes each objective.
	extremes := make([]*Solution, m)
	for k := range m {
		best := pf.Solutions[0]
		bestVal := best.Objectives[k]
		for _, s := range pf.Solutions[1:] {
			if s.Objectives[k] > bestVal {
				best = s
				bestVal = s.Objectives[k]
			}
		}
		extremes[k] = best
	}

	// If extremes collapse (e.g., one point maximizes everything), the hyperplane is degenerate.
	allSame := true
	for k := 1; k < m; k++ {
		if extremes[k] != extremes[0] {
			allSame = false
			break
		}
	}
	if allSame {
		return nil
	}

	// m == 3
	p0 := r3.Vec{X: norm(extremes[0].Objectives[0], 0), Y: norm(extremes[0].Objectives[1], 1), Z: norm(extremes[0].Objectives[2], 2)}
	p1 := r3.Vec{X: norm(extremes[1].Objectives[0], 0), Y: norm(extremes[1].Objectives[1], 1), Z: norm(extremes[1].Objectives[2], 2)}
	p2 := r3.Vec{X: norm(extremes[2].Objectives[0], 0), Y: norm(extremes[2].Objectives[1], 1), Z: norm(extremes[2].Objectives[2], 2)}

	v1 := r3.Sub(p1, p0)
	v2 := r3.Sub(p2, p0)
	nVec := r3.Cross(v1, v2)
	den := r3.Norm(nVec)
	if den == 0 {
		return nil
	}

	ideal := r3.Vec{X: 1, Y: 1, Z: 1}
	if r3.Dot(nVec, r3.Sub(ideal, p0)) < 0 {
		nVec = r3.Scale(-1, nVec)
	}

	best := pf.Solutions[0]
	bestDist := math.Inf(-1)
	for _, s := range pf.Solutions {
		x := r3.Vec{X: norm(s.Objectives[0], 0), Y: norm(s.Objectives[1], 1), Z: norm(s.Objectives[2], 2)}
		d := r3.Dot(nVec, r3.Sub(x, p0)) / den
		if d > bestDist {
			bestDist = d
			best = s
		}
	}

	return best.Copy()
}

// FindKneePoint returns a compromise solution on the Pareto front.
//
// Implementation:
//   - Primary: normalize objectives and pick the point with maximum distance
//     from the hyperplane spanned by extreme points.
//   - Fallback: if the hyperplane is degenerate, pick the point closest to the
//     ideal (utopia) point.
func (pf *ParetoFront) FindKneePoint() *Solution {
	if len(pf.Solutions) == 0 {
		return nil
	}

	if knee := pf.findKneePointByHyperplane(); knee != nil {
		return knee
	}
	return pf.findClosestToIdealPoint()
}

// Len returns the number of solutions in the Pareto front
func (pf *ParetoFront) Len() int {
	return len(pf.Solutions)
}
