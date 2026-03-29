package qlearning

import (
	"fmt"
	"math"
	"sort"
	"strings"
	"time"

	"github.com/HubertasVin/robust-facility-location/config"
	"github.com/HubertasVin/robust-facility-location/problem"
	"github.com/HubertasVin/robust-facility-location/rng"
)

// Agent encapsulates the Q-learning logic for facility placement.
type Agent struct {
	Cfg    *config.Config
	Prob   *problem.Problem
	QTable *QTable
}

// NewAgent creates a new Agent.
func NewAgent(cfg *config.Config, prob *problem.Problem) *Agent {
	return &Agent{
		Cfg:    cfg,
		Prob:   prob,
		QTable: NewQTable(),
	}
}

// AvailableActions returns candidate location indices not already in state.
func (a *Agent) AvailableActions(state []int) []int {
	inState := make(map[int]bool, len(state))
	for _, s := range state {
		inState[s] = true
	}
	avail := make([]int, 0, len(a.Prob.L))
	for _, loc := range a.Prob.L {
		if !inState[loc] {
			avail = append(avail, loc)
		}
	}
	return avail
}

// Utility is a convenience wrapper that computes binary utility for a state.
func (a *Agent) Utility(state []int) float64 {
	s := make([]int, len(state))
	copy(s, state)
	sort.Ints(s)
	return a.Prob.UtilityBinary(s)
}

// EpsilonValue computes a decaying epsilon for a given episode.
func (a *Agent) EpsilonValue(episode int) float64 {
	return 1.0 / (1.0 + float64(episode)/a.Cfg.EpsilonSlope)
}

// ActionEpsilonGreedy picks an action using epsilon-greedy policy.
func (a *Agent) ActionEpsilonGreedy(state []int, eps float64, qt *QTable) (action int, wasRandom bool) {
	if rng.Float64() < eps {
		avail := a.AvailableActions(state)
		if len(avail) == 0 {
			return -1, true
		}
		return avail[rng.Intn(len(avail))], true
	}
	return a.ActionGreedy(state, qt), false
}

// ActionGreedy picks the action with the highest Q-value.
func (a *Agent) ActionGreedy(state []int, qt *QTable) int {
	bestQ := math.Inf(-1)
	bestFacility := -1

	for _, loc := range a.AvailableActions(state) {
		key := StateActionKey(state, loc)
		if qt.Has(key) {
			q := qt.Get(key)
			if q > bestQ {
				bestQ = q
				bestFacility = loc
			}
		}
	}

	if bestFacility == -1 {
		// Fallback: random from all candidates
		return a.Prob.L[rng.Intn(len(a.Prob.L))]
	}
	return bestFacility
}

// MaxFutureQ returns the maximum Q-value reachable from nextState.
func (a *Agent) MaxFutureQ(nextState []int, qt *QTable) float64 {
	best := math.Inf(-1)
	for _, loc := range a.AvailableActions(nextState) {
		key := StateActionKey(nextState, loc)
		if qt.Has(key) {
			if v := qt.Get(key); v > best {
				best = v
			}
		}
	}
	if math.IsInf(best, -1) {
		return 0
	}
	return best
}

// UpdateQ performs a single Q-learning update.
func (a *Agent) UpdateQ(stateActionKey string, nextState []int, reward float64, qt *QTable) {
	qPredict := qt.Get(stateActionKey)
	maxFuture := a.MaxFutureQ(nextState, qt)
	qTarget := reward + a.Cfg.Gamma*maxFuture
	qt.Set(stateActionKey, qPredict+a.Cfg.Alpha*(qTarget-qPredict))
}

// Train runs the training loop for a given number of episodes on a local Q-table.
func (a *Agent) Train(episodes int, qt *QTable) {
	for ep := 0; ep < episodes; ep++ {
		eps := a.Cfg.Epsilon
		if !a.Cfg.FixedEpsilon {
			eps = a.EpsilonValue(ep)
		}

		if a.Cfg.UseFinalRewardQ {
			a.RunEpisodeFinalReward(eps, qt)
		} else {
			a.RunEpisodeIntermediateReward(eps, qt)
		}
	}
}

// GetOptimalSolution extracts the greedy solution from a Q-table.
func (a *Agent) GetOptimalSolution() []int {
	return a.GetOptimalSolutionFromQT(a.QTable)
}

// GetOptimalSolutionFromQT extracts the greedy solution from a specific Q-table.
func (a *Agent) GetOptimalSolutionFromQT(qt *QTable) []int {
	state := make([]int, 0, a.Cfg.MaxFacilities)
	for step := 0; step < a.Cfg.MaxFacilities; step++ {
		best := a.ActionGreedy(state, qt)
		if best == -1 {
			break
		}
		state = append(state, best)
		sort.Ints(state)
	}
	return state
}

// RunMultipleTrainingRuns performs multiple independent training runs and
// keeps the last run's Q-table in the agent's global QTable.
func (a *Agent) RunMultipleTrainingRuns() {
	fmt.Println("\n=== Training Run:")

	baseName := a.Cfg.QTableFile
	if idx := strings.LastIndex(baseName, "."); idx != -1 {
		baseName = baseName[:idx]
	}
	savePath := baseName + ".dat"

	for run := 0; run < a.Cfg.NumTrainingRuns; run++ {
		seed := uint64(time.Now().UnixNano()) + uint64(run)*1000
		rng.Seed(seed)

		localQT := NewQTable()

		fmt.Printf("%d (seed=%d)\n", run+1, seed)

		a.Train(a.Cfg.TrainingEpisodes, localQT)

		solution := a.GetOptimalSolutionFromQT(localQT)
		finalUtility := a.Utility(solution)
		fmt.Printf("Run %d final utility: %.6f\n", run, finalUtility)

		if err := localQT.Save(savePath); err != nil {
			fmt.Printf("Warning: failed to save Q-table for run %d: %v\n", run, err)
		}
	}

	// Load the last saved Q-table as the global one
	if err := a.QTable.Load(baseName + ".dat"); err != nil {
		fmt.Printf("Warning: failed to load final Q-table: %v\n", err)
	}

	fmt.Println(" ===")
}
