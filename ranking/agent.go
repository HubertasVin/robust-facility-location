package ranking

import (
	"fmt"
	"math"
	"sort"

	"github.com/HubertasVin/robust-facility-location/config"
	"github.com/HubertasVin/robust-facility-location/problem"
	"github.com/HubertasVin/robust-facility-location/rng"
)

// Agent implements the FLARC/PL algorithm for facility location.
type Agent struct {
	Cfg       *config.Config
	Prob      *problem.Problem
	RankTable *RankTable
	Behavior  problem.CustomerBehaviorModel
	baseline  float64
}

// NewAgent creates a new Agent.
func NewAgent(cfg *config.Config, prob *problem.Problem, behavior problem.CustomerBehaviorModel) *Agent {
	if behavior == nil {
		behavior = problem.BinaryModel{}
	}
	rt := NewRankTable()
	rt.Initialize(prob.L)
	return &Agent{
		Cfg:       cfg,
		Prob:      prob,
		RankTable: rt,
		Behavior:  behavior,
		baseline:  0.0,
	}
}

// Utility evaluates a solution's utility.
func (a *Agent) Utility(locations []int) float64 {
	return a.Behavior.Utility(a.Prob, locations)
}

// generateInitialSolution creates a random initial solution.
func (a *Agent) generateInitialSolution() *Individual {
	// Randomly select MaxFacilities locations
	n := len(a.Prob.L)
	perm := make([]int, n)
	for i := range perm {
		perm[i] = i
	}
	// Fisher-Yates shuffle
	for i := n - 1; i > 0; i-- {
		j := rng.Intn(i + 1)
		perm[i], perm[j] = perm[j], perm[i]
	}

	locations := make([]int, a.Cfg.MaxFacilities)
	for i := 0; i < a.Cfg.MaxFacilities; i++ {
		locations[i] = perm[i]
	}
	sort.Ints(locations)

	ind := &Individual{Locations: locations}
	ind.Utility = a.Utility(locations)
	return ind
}

// generateInitialSolutionFromRanks creates an initial solution biased by ranks.
func (a *Agent) generateInitialSolutionFromRanks() *Individual {
	// Calculate sampling probabilities based on ranks
	probs := a.calculateRankProbabilities(nil, -1)

	locations := make([]int, 0, a.Cfg.MaxFacilities)
	used := make(map[int]bool)

	for len(locations) < a.Cfg.MaxFacilities {
		// Sample a location based on probabilities
		loc := a.sampleLocation(probs, used)
		if loc == -1 {
			break
		}
		locations = append(locations, loc)
		used[loc] = true
	}

	sort.Ints(locations)
	ind := &Individual{Locations: locations}
	ind.Utility = a.Utility(locations)
	return ind
}

// calculateRankProbabilities computes sampling probabilities based on ranks.
// If changingLoc >= 0, probabilities are weighted by inverse distance to that location.
func (a *Agent) calculateRankProbabilities(exclude map[int]bool, changingLoc int) []float64 {
	n := len(a.Prob.L)
	probs := make([]float64, n)

	// Get min and max ranks
	minR, maxR := a.RankTable.MinMax()
	rangeR := maxR - minR
	if rangeR == 0 {
		rangeR = 1.0 // Avoid division by zero
	}

	// Calculate normalized ranks
	normalizedRanks := make([]float64, n)
	for i := range n {
		loc := a.Prob.L[i]
		r := a.RankTable.Get(loc)
		normalizedRanks[i] = (r - minR) / rangeR
	}

	// Find max normalized rank for softmax stability
	maxNorm := 0.0
	for _, r := range normalizedRanks {
		if r > maxNorm {
			maxNorm = r
		}
	}

	// Calculate softmax with max shift
	sumZ := 0.0
	for i := range n {
		if exclude != nil && exclude[a.Prob.L[i]] {
			continue
		}
		expVal := math.Exp(normalizedRanks[i] - maxNorm)

		// Weight by inverse distance if changing a specific location
		if changingLoc >= 0 {
			dist := a.Prob.Distance(a.Prob.L[i], a.Prob.L[changingLoc])
			if dist > 0 {
				expVal /= dist
			}
		}

		probs[i] = expVal
		sumZ += expVal
	}

	// Normalize probabilities
	if sumZ > 0 {
		for i := range probs {
			probs[i] /= sumZ
		}
	}

	// Set excluded locations to 0
	if exclude != nil {
		for i := range n {
			if exclude[a.Prob.L[i]] {
				probs[i] = 0
			}
		}
	}

	return probs
}

// sampleLocation samples a location index based on probabilities.
func (a *Agent) sampleLocation(probs []float64, exclude map[int]bool) int {
	// Recalculate sum excluding already used locations
	sum := 0.0
	for i, p := range probs {
		if exclude != nil && exclude[a.Prob.L[i]] {
			continue
		}
		sum += p
	}

	if sum == 0 {
		return -1
	}

	r := rng.Float64() * sum
	cumulative := 0.0
	for i, p := range probs {
		if exclude != nil && exclude[a.Prob.L[i]] {
			continue
		}
		cumulative += p
		if r <= cumulative {
			return a.Prob.L[i]
		}
	}

	// Fallback: return last non-excluded location
	for i := len(probs) - 1; i >= 0; i-- {
		if exclude == nil || !exclude[a.Prob.L[i]] {
			return a.Prob.L[i]
		}
	}
	return -1
}

// mutate creates a new solution by potentially changing one location.
func (a *Agent) mutate(parent *Individual) *Individual {
	child := parent.Copy()

	// With probability epsilon, change a random location
	if rng.Float64() < a.Cfg.Epsilon {
		// Select random position to change
		pos := rng.Intn(len(child.Locations))
		oldLoc := child.Locations[pos]

		// Build exclusion set (current solution locations)
		exclude := make(map[int]bool)
		for _, loc := range child.Locations {
			exclude[loc] = true
		}

		// Find index of oldLoc in L
		oldLocIdx := -1
		for i, l := range a.Prob.L {
			if l == oldLoc {
				oldLocIdx = i
				break
			}
		}

		// Calculate probabilities weighted by distance to the location being changed
		probs := a.calculateRankProbabilities(exclude, oldLocIdx)

		// Sample new location
		newLoc := a.sampleLocation(probs, exclude)
		if newLoc != -1 && newLoc != oldLoc {
			child.Locations[pos] = newLoc
			sort.Ints(child.Locations)
		}
	}

	child.Utility = a.Utility(child.Locations)
	return child
}

// updateRanks updates rank scores for locations in the solution.
func (a *Agent) updateRanks(solution *Individual) {
	reward := solution.Utility - a.baseline
	for _, loc := range solution.Locations {
		delta := a.Cfg.Alpha * reward
		a.RankTable.Update(loc, delta)
	}
}

// updateBaseline updates the moving average baseline.
func (a *Agent) updateBaseline(utility float64) {
	a.baseline = (1-a.Cfg.Alpha)*a.baseline + a.Cfg.Alpha*utility
}

// Run executes the FLARC/PL algorithm.
func (a *Agent) Run() *Individual {
	pop := NewPopulation(a.Cfg.PopulationSize)

	// Initialize population with random solutions
	// Use rank-biased initialization if we have prior knowledge
	hasRanks := a.RankTable.Len() > 0
	for pop.Len() < a.Cfg.PopulationSize {
		var ind *Individual
		if hasRanks && rng.Float64() < 0.5 {
			ind = a.generateInitialSolutionFromRanks()
		} else {
			ind = a.generateInitialSolution()
		}
		pop.Add(ind)
	}

	// Initialize baseline from population average
	a.baseline = pop.AverageUtility()

	bestEver := pop.Best().Copy()

	// Main loop
	for iter := 0; iter < a.Cfg.Iterations; iter++ {
		// Select random solution from population
		parent := pop.RandomSelect()

		// Create offspring through mutation
		child := a.mutate(parent)

		// Update ranks based on child performance
		a.updateRanks(child)

		// Update baseline
		a.updateBaseline(child.Utility)

		// Try to add child to population
		pop.Add(child)

		// Track best solution ever found
		if child.Utility > bestEver.Utility {
			bestEver = child.Copy()
		}

		// Progress reporting
		if (iter+1)%(a.Cfg.Iterations/10) == 0 {
			fmt.Printf("Iteration %d: Best=%.6f%%, PopBest=%.6f%%, Baseline=%.6f%%\n",
				iter+1, bestEver.Utility, pop.Best().Utility, a.baseline)
		}
	}

	return bestEver
}

// GetOptimalSolution returns the best solution found using current ranks.
func (a *Agent) GetOptimalSolution() []int {
	// Generate multiple solutions and return the best
	best := a.generateInitialSolutionFromRanks()
	for range 100 {
		candidate := a.generateInitialSolutionFromRanks()
		if candidate.Utility > best.Utility {
			best = candidate
		}
	}
	return best.Locations
}
