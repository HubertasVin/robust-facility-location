package solution

import (
	"fmt"
	"strings"
)

// Solution represents a facility location solution.
type Solution struct {
	Locations     []int
	Objectives    []float64
	DominanceRank int
}

// NewSolution creates a Solution with pre-allocated slices.
func NewSolution(numLocations, numObjectives int) *Solution {
	return &Solution{
		Locations:  make([]int, numLocations),
		Objectives: make([]float64, numObjectives),
	}
}

// String returns a human-readable representation.
func (s *Solution) String() string {
	var b strings.Builder
	for _, loc := range s.Locations {
		fmt.Fprintf(&b, "%d\t", loc)
	}
	for _, obj := range s.Objectives {
		fmt.Fprintf(&b, "%.6f\t", obj)
	}
	fmt.Fprintf(&b, "%d", s.DominanceRank)
	return b.String()
}
