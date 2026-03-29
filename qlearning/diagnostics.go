package qlearning

import (
	"fmt"
	"sort"
)

// MarginalGain pairs an action with its true marginal utility gain.
type MarginalGain struct {
	Action int
	Gain   float64
}

// MarginalGains computes, for each available action from a state,
// the true utility gain r(a) = U(s ∪ {a}) - U(s).
// Results are sorted descending by gain.
func (a *Agent) MarginalGains(state []int) []MarginalGain {
	uBefore := a.Utility(state)

	gains := make([]MarginalGain, 0, len(a.Prob.L))
	for _, action := range a.AvailableActions(state) {
		nextState := make([]int, len(state)+1)
		copy(nextState, state)
		nextState[len(state)] = action
		sort.Ints(nextState)

		uAfter := a.Utility(nextState)
		gains = append(gains, MarginalGain{
			Action: action,
			Gain:   uAfter - uBefore,
		})
	}

	sort.Slice(gains, func(i, j int) bool {
		return gains[i].Gain > gains[j].Gain
	})

	return gains
}

// GetQ returns the Q-value for a (state, action) pair from the agent's global Q-table.
func (a *Agent) GetQ(state []int, action int) float64 {
	key := StateActionKey(state, action)
	return a.QTable.Get(key)
}

// MaxFutureQExistingKeys returns the maximum Q-value over all available
// actions from the given state, considering only keys that exist in the table.
func (a *Agent) MaxFutureQExistingKeys(state []int) float64 {
	return a.MaxFutureQ(state, a.QTable)
}

// DiagnoseStep prints a detailed comparison of Q-values vs. true marginal
// gains for every available action from a given state. Useful for debugging
// whether the Q-table has converged to reflect true utilities.
func (a *Agent) DiagnoseStep(state []int) {
	fmt.Printf("--- Diagnosis for state %v (U=%.4f) ---\n", state, a.Utility(state))

	gains := a.MarginalGains(state)

	fmt.Printf("%-10s %-14s %-14s %-10s\n", "Action", "MarginalGain", "Q-Value", "Match?")
	fmt.Println("----------------------------------------------")

	for _, mg := range gains {
		qVal := a.GetQ(state, mg.Action)
		match := ""
		// Check if Q-value ranking agrees with true gain ranking
		if qVal > 0 && mg.Gain > 0 {
			match = "✓"
		} else if qVal <= 0 && mg.Gain <= 0 {
			match = "✓"
		} else {
			match = "✗"
		}
		fmt.Printf("%-10d %-14.6f %-14.6f %-10s\n", mg.Action, mg.Gain, qVal, match)
	}
	fmt.Println()
}

// DiagnoseFullSolution runs DiagnoseStep at every step of the greedy solution
// extraction, showing how the agent builds its solution.
func (a *Agent) DiagnoseFullSolution() {
	state := make([]int, 0, a.Cfg.MaxFacilities)
	fmt.Println("========== Full Solution Diagnosis ==========")

	for step := 0; step < a.Cfg.MaxFacilities; step++ {
		a.DiagnoseStep(state)

		best := a.ActionGreedy(state, a.QTable)
		if best == -1 {
			fmt.Println("No more actions available.")
			break
		}
		fmt.Printf(">> Step %d: selected action %d\n\n", step+1, best)
		state = append(state, best)
		sort.Ints(state)
	}

	fmt.Printf("Final solution: %v (U=%.6f%%)\n", state, a.Utility(state))
	fmt.Println("=============================================")
}
