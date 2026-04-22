package main

import (
	"fmt"
	"log"
	"time"

	"github.com/HubertasVin/robust-facility-location/config"
	"github.com/HubertasVin/robust-facility-location/problem"
	"github.com/HubertasVin/robust-facility-location/ranking"
	"github.com/HubertasVin/robust-facility-location/rng"
)

func main() {
	cfg := config.Load()

	rng.Seed(uint64(time.Now().UnixNano()))

	prob, err := problem.LoadProblem(cfg.ProblemFile, cfg.DemandsFile)
	if err != nil {
		log.Fatalf("Failed to load problem data: %v", err)
	}

	behaviorModel := problem.HuffModel{}
	// Example alternatives:
	// behaviorModel := problem.PartiallyBinaryModel{}
	// behaviorModel := problem.ParetoHuffModel{}
	// behaviorModel := problem.UtilityFunc(func(p *problem.Problem, x []int) float64 {
	// 	return problem.ParetoHuffModel{}.Utility(p, x)
	// })

	agent := ranking.NewAgent(cfg, prob, behaviorModel)

	// Try to load existing ranks (transfers experience across instances)
	_ = agent.RankTable.Load(cfg.RankFile)

	if cfg.PerformTraining {
		fmt.Println("\n=== Running FLARC/PL Optimization ===")
		best := agent.Run()

		if err := agent.RankTable.Save(cfg.RankFile); err != nil {
			log.Fatalf("Failed to save rank table: %v", err)
		}

		fmt.Printf("\nBest solution found: ")
		for _, loc := range best.Locations {
			fmt.Printf("%d ", loc)
		}
		fmt.Printf("(%.6f%%)\n", best.Utility)
	} else {
		optimalSolution := agent.GetOptimalSolution()
		utility := behaviorModel.Utility(prob, optimalSolution)

		fmt.Printf("Optimal locations for the new facilities: ")
		for _, loc := range optimalSolution {
			fmt.Printf("%d ", loc)
		}
		fmt.Printf("(%.6f%%)\n", utility)
	}
}
