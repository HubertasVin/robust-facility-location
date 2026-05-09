package ranking

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
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
	Behaviour  problem.CustomerBehaviourModel
	Logger    EvaluationLogger
	// LogBehaviours, when set, controls which objectives are logged for each evaluated solution.
	// If nil/empty, logging (if enabled) falls back to only the active Behaviour.
	LogBehaviours []problem.CustomerBehaviourModel
	// ParetoFront is populated after FindRobustSolution is called
	ParetoFront *solution.ParetoFront
	baseline    float64
}

// NewAgent creates a new Agent.
func NewAgent(cfg *config.Config, prob *problem.Problem, behaviour problem.CustomerBehaviourModel) *Agent {
	if behaviour == nil {
		behaviour = problem.BinaryModel{}
	}
	rt := NewRankTable()
	rt.Initialize(prob.L)
	return &Agent{
		Cfg:          cfg,
		Prob:         prob,
		RankTable:    rt,
		Behaviour:     behaviour,
		Logger:       nil,
		LogBehaviours: nil,
		baseline:     0.0,
	}
}

func (a *Agent) log(stage string, iter int, locations []int, behaviours []problem.CustomerBehaviourModel, objectives []float64) {
	if a.Logger == nil {
		return
	}
	_ = a.Logger.Record(stage, iter, locations, behaviours, objectives)
}

// Utility evaluates a solution's utility.
func (a *Agent) Utility(locations []int) float64 {
	return a.Behaviour.Utility(a.Prob, locations)
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

	// Calculate softmax weights. Excluded candidates are assigned weight 0.
	for i := range n {
		locID := a.Prob.L[i]
		if exclude != nil && exclude[locID] {
			probs[i] = 0
			continue
		}
		expVal := math.Exp(normalizedRanks[i])
		if changingLocID >= 0 {
			dist := a.Prob.Distance(locID, changingLocID)
			if dist > 0 {
				expVal /= dist
			}
		}
		probs[i] = expVal

	}
	
	// Normalize probabilities (probs[i] /= sumProbs).
	sumProbs := floats.Sum(probs)
	if sumProbs > 0 {
		floats.Scale(1.0/sumProbs, probs)
	}

	return probs
}

// sampleLocation samples and returns a candidate facility location ID (an element of L)
// using the provided weights.
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
	behaviours := a.LogBehaviours
	if len(behaviours) == 0 {
		behaviours = []problem.CustomerBehaviourModel{a.Behaviour}
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

		// Log objectives for the agent's behaviour (or more, if the caller supplies more in robust mode).
		objectives := a.evaluateMultiObjective(ind.Locations, behaviours)
		a.log("train:init", -1, ind.Locations, behaviours, objectives)

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
		objectives := a.evaluateMultiObjective(child.Locations, behaviours)
		a.log("train:iter", iter, child.Locations, behaviours, objectives)

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
			fmt.Fprintf(os.Stderr, "Iteration %d: Best=%.6f%%, PopBest=%.6f%%, Baseline=%.6f%%\n",
				iter+1, bestEver.Utility, pop.Best().Utility, a.baseline)
		}
	}

	return bestEver
}

// evaluateMultiObjective evaluates a solution against multiple customer behaviour models
func (a *Agent) evaluateMultiObjective(locations []int, behaviours []problem.CustomerBehaviourModel) []float64 {
	objectives := make([]float64, len(behaviours))
	for i, behaviour := range behaviours {
		objectives[i] = behaviour.Utility(a.Prob, locations)
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
func (a *Agent) FindRobustSolution(behaviours []problem.CustomerBehaviourModel) *solution.Solution {
	if len(behaviours) == 0 {
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

		objectives := a.evaluateMultiObjective(ind.Locations, behaviours)
		ind.Utility = stat.Mean(objectives, nil)
		a.log("robust:init", -1, ind.Locations, behaviours, objectives)
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

	snapEvery := a.Cfg.LogPeriod

	// Main loop: mutate, re-rank, and accumulate non-dominated solutions.
	for iter := 0; iter < a.Cfg.Iterations; iter++ {
		parent := pop.RandomSelect()
		child := a.mutate(parent)

		objectives := a.evaluateMultiObjective(child.Locations, behaviours)
		child.Utility = stat.Mean(objectives, nil)
		a.log("robust:iter", iter, child.Locations, behaviours, objectives)

		a.updateRanks(child)
		a.updateBaseline(child.Utility)
		pop.Add(child)
		paretoFront.AddSolution(a.createMultiObjectiveSolutionWithObjectives(child.Locations, objectives))

		if child.Utility > bestMean.Utility {
			bestMean = child.Copy()
		}

		if snapEvery > 0 && (iter+1)%snapEvery == 0 {
			snapKP := paretoFront.FindKneePoint()
			if snapKP != nil {
				snap := struct {
					Iteration  int       `json:"iteration"`
					ParetoSize int       `json:"pareto_size"`
					KneePoint  []float64 `json:"knee_point"`
				}{
					Iteration:  iter + 1,
					ParetoSize: paretoFront.Len(),
					KneePoint:  snapKP.Objectives,
				}
				enc := json.NewEncoder(os.Stderr)
				_ = enc.Encode(snap)
			}
		}

		if snapEvery <= 0 && (iter+1)%reportEvery == 0 {
			fmt.Fprintf(os.Stderr, "Iteration %d: Pareto=%d, BestMean=%.6f%%, Baseline=%.6f%%\n",
				iter+1, paretoFront.Len(), bestMean.Utility, a.baseline)
		}
	}

	a.ParetoFront = paretoFront

	kneePoint := paretoFront.FindKneePoint()
	if kneePoint == nil {
		return nil
	}

	fmt.Fprintf(os.Stderr, "Pareto front contains %d solutions\n", paretoFront.Len())
	fmt.Fprintf(os.Stderr, "Knee point objectives: %v\n", kneePoint.Objectives)

	return kneePoint
}
