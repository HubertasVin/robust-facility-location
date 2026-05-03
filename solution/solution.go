package solution

import (
	"math"

	"gonum.org/v1/gonum/floats"
)

// Solution represents a facility location solution.
type Solution struct {
	Locations  []int     `json:"locations"`
	Objectives []float64 `json:"objectives"`
}

// NewSolution creates a Solution with pre-allocated slices.
func NewSolution(numLocations, numObjectives int) *Solution {
	return &Solution{
		Locations:  make([]int, numLocations),
		Objectives: make([]float64, numObjectives),
	}
}

// Dominates checks if this solution dominates another solution.
// A solution dominates another if it is at least as good in all objectives
// and strictly better in at least one objective.
func (s *Solution) Dominates(other *Solution) bool {
	if len(s.Objectives) != len(other.Objectives) {
		return false
	}

	atLeastOneBetter := false
	for i := range s.Objectives {
		// Assuming all objectives are to be maximized
		if s.Objectives[i] < other.Objectives[i] {
			return false
		}
		if s.Objectives[i] > other.Objectives[i] {
			atLeastOneBetter = true
		}
	}
	return atLeastOneBetter
}

// CalculateDistanceToIdeal calculates the Euclidean distance to an ideal point.
// The ideal point has the maximum value for each objective.
func (s *Solution) CalculateDistanceToIdeal(idealPoint []float64) float64 {
	if len(s.Objectives) != len(idealPoint) {
		return math.Inf(1)
	}
	// Euclidean distance (L2 norm).
	return floats.Distance(idealPoint, s.Objectives, 2)
}

// Copy creates a deep copy of the solution.
func (s *Solution) Copy() *Solution {
	locations := make([]int, len(s.Locations))
	copy(locations, s.Locations)

	objectives := make([]float64, len(s.Objectives))
	copy(objectives, s.Objectives)

	return &Solution{
		Locations:  locations,
		Objectives: objectives,
	}
}
