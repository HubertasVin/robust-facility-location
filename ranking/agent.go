package ranking

import (
	"fmt"
	"math"
	"math/rand/v2"
	"sort"

	"github.com/HubertasVin/robust-facility-location/config"
	"github.com/HubertasVin/robust-facility-location/problem"
	"github.com/HubertasVin/robust-facility-location/solution"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/stat"
)

// Agent implements the FLARC/PL algorithm for facility location.
type Agent struct {
	Cfg       *config.Config
	Prob      *problem.Problem
	RankTable *RankTable
	Behavior  problem.CustomerBehaviorModel
	Logger    EvaluationLogger
	// LogBehaviors, when set, controls which objectives are logged for each evaluated solution.
	// If nil/empty, logging (if enabled) falls back to only the active Behavior.
	LogBehaviors []problem.CustomerBehaviorModel
	baseline     float64
}

// NewAgent creates a new Agent.
func NewAgent(cfg *config.Config, prob *problem.Problem, behavior problem.CustomerBehaviorModel) *Agent {
	if behavior == nil {
		behavior = problem.BinaryModel{}
	}
	rt := NewRankTable()
	rt.Initialize(prob.L)
	return &Agent{
		Cfg:          cfg,
		Prob:         prob,
		RankTable:    rt,
		Behavior:     behavior,
		Logger:       nil,
		LogBehaviors: nil,
		baseline:     0.0,
	}
}

func (a *Agent) log(stage string, iter int, locations []int, behaviors []problem.CustomerBehaviorModel, objectives []float64) {
	if a.Logger == nil {
		return
	}
	_ = a.Logger.Record(stage, iter, locations, behaviors, objectives)
}

// Utility evaluates a solution's utility.
func (a *Agent) Utility(locations []int) float64 {
	return a.Behavior.Utility(a.Prob, locations)
}

// generateInitialSolution creates a random initial solution.
func (a *Agent) generateInitialSolution() *Individual {
	// Randomly select MaxFacilities locations (facility IDs from L)
	n := len(a.Prob.L)
	perm := make([]int, n)
	copy(perm, a.Prob.L)
	rand.Shuffle(n, func(i, j int) {
		perm[i], perm[j] = perm[j], perm[i]
	})

	maxFacilities := a.Cfg.MaxFacilities
	if maxFacilities > n {
		maxFacilities = n
	}
	locations := make([]int, maxFacilities)
	copy(locations, perm[:maxFacilities])
	sort.Ints(locations)

	ind := &Individual{Locations: locations}
	ind.Utility = a.Utility(locations)
	return ind
}

// generateInitialSolutionFromRanks creates an initial solution biased by ranks.
func (a *Agent) generateInitialSolutionFromRanks() *Individual {
	locations := make([]int, 0, a.Cfg.MaxFacilities)
	used := make(map[int]bool)

	for len(locations) < a.Cfg.MaxFacilities {
		// Calculate sampling probabilities based on ranks, excluding already used IDs.
		probs := a.calculateRankProbabilities(used, -1)

		// Sample a location based on probabilities
		loc := a.sampleLocation(probs)
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
// If changingLocID >= 0, probabilities are weighted by inverse distance to that facility ID.
func (a *Agent) calculateRankProbabilities(exclude map[int]bool, changingLocID int) []float64 {
	n := len(a.Prob.L)
	probs := make([]float64, n)
	if n == 0 {
		return probs
	}

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

	// Find max normalized rank for softmax stability.
	maxNorm := floats.Max(normalizedRanks)

	// Calculate softmax weights (max-shifted). Excluded candidates are assigned weight 0.
	for i := range n {
		locID := a.Prob.L[i]
		if exclude != nil && exclude[locID] {
			probs[i] = 0
			continue
		}
		expVal := math.Exp(normalizedRanks[i] - maxNorm)
		if changingLocID >= 0 {
			dist := a.Prob.Distance(locID, changingLocID)
			if dist > 0 {
				expVal /= dist
			}
		}
		probs[i] = expVal
	}

	// Normalize probabilities.
	sumZ := floats.Sum(probs)
	if sumZ > 0 {
		floats.Scale(1.0/sumZ, probs)
	}

	return probs
}

// sampleLocation samples and returns a candidate facility location ID (an element of L)
// using the provided weights.
//
// probByIndex must be aligned with a.Prob.L. To exclude candidates, set their
// weights to 0 before calling.
func (a *Agent) sampleLocation(probByIndex []float64) int {
	if len(probByIndex) != len(a.Prob.L) {
		return -1
	}

	sum := floats.Sum(probByIndex)
	if sum == 0 {
		return -1
	}

	r := rand.Float64() * sum
	cumulative := 0.0
	for i, w := range probByIndex {
		cumulative += w
		if r <= cumulative {
			return a.Prob.L[i]
		}
	}

	// Fallback: return last candidate with non-zero weight.
	for i := len(probByIndex) - 1; i >= 0; i-- {
		if probByIndex[i] > 0 {
			return a.Prob.L[i]
		}
	}
	return -1
}

// mutate creates a new solution by potentially changing one location.
func (a *Agent) mutate(parent *Individual) *Individual {
	child := parent.Copy()

	// With probability epsilon, change a random location
	if rand.Float64() < a.Cfg.Epsilon {
		// Select random position to change
		pos := rand.IntN(len(child.Locations))
		oldLoc := child.Locations[pos]

		// Build exclusion set (current solution locations)
		exclude := make(map[int]bool)
		for _, loc := range child.Locations {
			exclude[loc] = true
		}

		// Calculate probabilities weighted by distance to the location being changed
		probs := a.calculateRankProbabilities(exclude, oldLoc)

		// Sample new location
		newLoc := a.sampleLocation(probs)
		if newLoc != -1 && newLoc != oldLoc {
			child.Locations[pos] = newLoc
			sort.Ints(child.Locations)
		}
	}

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
	behaviors := a.LogBehaviors
	if len(behaviors) == 0 {
		behaviors = []problem.CustomerBehaviorModel{a.Behavior}
	}

	// Initialize population with random solutions
	// Use rank-biased initialization if we have prior knowledge
	hasRanks := a.RankTable.Len() > 0
	for pop.Len() < a.Cfg.PopulationSize {
		var ind *Individual
		if hasRanks && rand.Float64() < 0.5 {
			ind = a.generateInitialSolutionFromRanks()
		} else {
			ind = a.generateInitialSolution()
		}

		// Log objectives for the agent's behavior (or more, if the caller supplies more in robust mode).
		objectives := a.evaluateMultiObjective(ind.Locations, behaviors)
		a.log("train:init", -1, ind.Locations, behaviors, objectives)

		pop.Add(ind)
	}

	// Initialize baseline from population average
	a.baseline = pop.AverageUtility()

	bestEver := pop.Best().Copy()

	// Main loop
	reportEvery := a.Cfg.Iterations / 10
	if reportEvery < 1 {
		reportEvery = 1
	}
	for iter := 0; iter < a.Cfg.Iterations; iter++ {
		// Select random solution from population
		parent := pop.RandomSelect()

		// Create offspring through mutation
		child := a.mutate(parent)
		child.Utility = a.Utility(child.Locations)
		objectives := a.evaluateMultiObjective(child.Locations, behaviors)
		a.log("train:iter", iter, child.Locations, behaviors, objectives)

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
		if (iter+1)%reportEvery == 0 {
			fmt.Printf("Iteration %d: Best=%.6f%%, PopBest=%.6f%%, Baseline=%.6f%%\n",
				iter+1, bestEver.Utility, pop.Best().Utility, a.baseline)
		}
	}

	return bestEver
}

// evaluateMultiObjective evaluates a solution against multiple customer behavior models
func (a *Agent) evaluateMultiObjective(locations []int, behaviors []problem.CustomerBehaviorModel) []float64 {
	objectives := make([]float64, len(behaviors))
	for i, behavior := range behaviors {
		objectives[i] = behavior.Utility(a.Prob, locations)
	}
	return objectives
}

func (a *Agent) createMultiObjectiveSolutionWithObjectives(locations []int, objectives []float64) *solution.Solution {
	sortedLocs := make([]int, len(locations))
	copy(sortedLocs, locations)
	sort.Ints(sortedLocs)

	sol := solution.NewSolution(len(sortedLocs), len(objectives))
	copy(sol.Locations, sortedLocs)
	copy(sol.Objectives, objectives)
	return sol
}

// FindRobustSolution finds a robust solution using knee point identification
func (a *Agent) FindRobustSolution(behaviors []problem.CustomerBehaviorModel) *solution.Solution {
	if len(behaviors) == 0 {
		return nil
	}

	pop := NewPopulation(a.Cfg.PopulationSize)
	paretoFront := solution.NewParetoFront()

	// Initialize population (utility = mean across objectives).
	for pop.Len() < a.Cfg.PopulationSize {
		var ind *Individual
		if rand.Float64() < 0.5 {
			ind = a.generateInitialSolutionFromRanks()
		} else {
			ind = a.generateInitialSolution()
		}

		objectives := a.evaluateMultiObjective(ind.Locations, behaviors)
		ind.Utility = stat.Mean(objectives, nil)
		a.log("robust:init", -1, ind.Locations, behaviors, objectives)
		pop.Add(ind)
		paretoFront.AddSolution(a.createMultiObjectiveSolutionWithObjectives(ind.Locations, objectives))
	}

	// Initialize baseline from population average (mean-objective score).
	a.baseline = pop.AverageUtility()
	bestMean := pop.Best().Copy()

	reportEvery := a.Cfg.Iterations / 10
	if reportEvery < 1 {
		reportEvery = 1
	}

	// Main loop: mutate, re-rank, and accumulate non-dominated solutions.
	for iter := 0; iter < a.Cfg.Iterations; iter++ {
		parent := pop.RandomSelect()
		child := a.mutate(parent)

		objectives := a.evaluateMultiObjective(child.Locations, behaviors)
		child.Utility = stat.Mean(objectives, nil)
		a.log("robust:iter", iter, child.Locations, behaviors, objectives)

		a.updateRanks(child)
		a.updateBaseline(child.Utility)
		pop.Add(child)
		paretoFront.AddSolution(a.createMultiObjectiveSolutionWithObjectives(child.Locations, objectives))

		if child.Utility > bestMean.Utility {
			bestMean = child.Copy()
		}

		if (iter+1)%reportEvery == 0 {
			fmt.Printf("Iteration %d: Pareto=%d, BestMean=%.6f%%, Baseline=%.6f%%\n",
				iter+1, paretoFront.Len(), bestMean.Utility, a.baseline)
		}
	}

	kneePoint := paretoFront.FindKneePoint()
	if kneePoint == nil {
		return nil
	}

	fmt.Printf("Pareto front contains %d solutions\n", paretoFront.Len())
	fmt.Printf("Knee point objectives: %v\n", kneePoint.Objectives)

	return kneePoint
}
