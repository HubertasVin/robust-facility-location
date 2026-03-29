package qlearning

import "sort"

// TrajectoryStep records a single step in an episode for deferred updates.
type TrajectoryStep struct {
	StateActionKey string
	NextStateKey   string
	NextState      []int
}

// RunEpisodeFinalReward runs one episode, collecting a trajectory, then
// updates Q-values using the final utility as the reward for every step.
func (a *Agent) RunEpisodeFinalReward(eps float64, qt *QTable) []int {
	state := make([]int, 0, a.Cfg.MaxFacilities)
	trajectory := make([]TrajectoryStep, 0, a.Cfg.MaxFacilities)

	for step := 0; step < a.Cfg.MaxFacilities; step++ {
		action, _ := a.ActionEpsilonGreedy(state, eps, qt)

		nextState := make([]int, len(state)+1)
		copy(nextState, state)
		nextState[len(state)] = action
		sort.Ints(nextState)

		trajectory = append(trajectory, TrajectoryStep{
			StateActionKey: StateActionKey(state, action),
			NextStateKey:   StateKey(nextState),
			NextState:      nextState,
		})

		state = nextState
	}

	// Deferred update with final utility
	finalUtility := a.Utility(state)
	for _, ts := range trajectory {
		qPredict := qt.Get(ts.StateActionKey)
		maxFuture := a.MaxFutureQ(ts.NextState, qt)
		qTarget := finalUtility + a.Cfg.Gamma*maxFuture

		// Only update if improving (matches original C++ behavior)
		if qTarget > qPredict {
			qt.Set(ts.StateActionKey, qPredict+a.Cfg.Alpha*(qTarget-qPredict))
		}
	}

	return state
}

// RunEpisodeIntermediateReward runs one episode with per-step reward updates.
func (a *Agent) RunEpisodeIntermediateReward(eps float64, qt *QTable) []int {
	state := make([]int, 0, a.Cfg.MaxFacilities)

	for step := 0; step < a.Cfg.MaxFacilities; step++ {
		action, _ := a.ActionEpsilonGreedy(state, eps, qt)

		nextState := make([]int, len(state)+1)
		copy(nextState, state)
		nextState[len(state)] = action
		sort.Ints(nextState)

		currentUtility := a.Utility(state)
		nextUtility := a.Utility(nextState)
		reward := nextUtility - currentUtility

		saKey := StateActionKey(state, action)
		a.UpdateQ(saKey, nextState, reward, qt)

		state = nextState
	}

	return state
}
