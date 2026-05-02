package ranking

import "github.com/HubertasVin/robust-facility-location/problem"

// EvaluationLogger receives every evaluated solution along with its objective
// values for a given set of behavior models.
//
// Implementations should be fast; callers may invoke this per-iteration.
//
// objectives are aligned with the behaviors slice that was used for evaluation.
type EvaluationLogger interface {
	Record(stage string, iter int, locations []int, behaviors []problem.CustomerBehaviorModel, objectives []float64) error
}
