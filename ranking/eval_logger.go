package ranking

import "github.com/HubertasVin/robust-facility-location/problem"

type EvaluationLogger interface {
	Record(stage string, iter int, locations []int, behaviours []problem.CustomerBehaviourModel, objectives []float64) error
}
