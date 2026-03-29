package main

import (
	"fmt"
	"log"
	"time"

	"github.com/HubertasVin/robust-facility-location/config"
	"github.com/HubertasVin/robust-facility-location/problem"
	"github.com/HubertasVin/robust-facility-location/qlearning"
	"github.com/HubertasVin/robust-facility-location/rng"
)

func main() {
	cfg := config.Load()

	rng.Seed(uint64(time.Now().UnixNano()))

	prob, err := problem.LoadProblem(cfg.ProblemFile, cfg.DemandsFile)
	if err != nil {
		log.Fatalf("Failed to load problem data: %v", err)
	}

	agent := qlearning.NewAgent(cfg, prob)

	if cfg.PerformTraining {
		agent.RunMultipleTrainingRuns()
		if err := agent.QTable.Save(cfg.QTableFile); err != nil {
			log.Fatalf("Failed to save Q-table: %v", err)
		}
	} else {
		if err := agent.QTable.Load(cfg.QTableFile); err != nil {
			log.Fatalf("Failed to load Q-table: %v", err)
		}
	}

	optimalSolution := agent.GetOptimalSolution()
	utility := prob.UtilityBinary(optimalSolution)

	fmt.Printf("Optimal locations for the new facilities: ")
	for _, loc := range optimalSolution {
		fmt.Printf("%d ", loc)
	}
	fmt.Printf("(%.6f%%)\n", utility)
}
