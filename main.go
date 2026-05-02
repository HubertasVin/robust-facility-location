package main

import (
	"fmt"
	"log"

	"github.com/HubertasVin/robust-facility-location/config"
	"github.com/HubertasVin/robust-facility-location/problem"
	"github.com/HubertasVin/robust-facility-location/ranking"
	"github.com/HubertasVin/robust-facility-location/report"
)

// AllBehaviorModels returns all available customer behavior models
func AllBehaviorModels() []problem.CustomerBehaviorModel {
	return []problem.CustomerBehaviorModel{
		problem.HuffModel{},
		problem.PartiallyBinaryModel{},
		problem.ParetoHuffModel{},
	}
}

func main() {
	cfg := config.Load()

	prob, err := problem.LoadProblem(cfg.ProblemFile, cfg.DemandsFile)
	if err != nil {
		log.Fatalf("Failed to load problem data: %v", err)
	}

	behaviorModel := problem.HuffModel{}
	// Example alternatives:
	// behaviorModel := problem.PartiallyBinaryModel{}
	// behaviorModel := problem.ParetoHuffModel{}

	agent := ranking.NewAgent(cfg, prob, behaviorModel)

	// Write a table of every evaluated solution (locations + objectives per behavior model).
	behaviors := AllBehaviorModels()
	tableLogger, err := report.NewSolutionsTableLogger(cfg.CheckedSolutionsFile, behaviors)
	if err != nil {
		log.Fatalf("Failed to create checked-solutions table file: %v", err)
	}
	defer func() {
		if err := tableLogger.Close(); err != nil {
			log.Printf("Warning: failed to close checked-solutions table: %v", err)
		}
	}()
	agent.Logger = tableLogger
	agent.LogBehaviors = behaviors

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
		// Find robust solution using knee point identification
		fmt.Println("\n=== Finding Robust Solution (Knee Point) ===")
		robustSolution := agent.FindRobustSolution(behaviors)
		if robustSolution == nil {
			log.Fatalf("Failed to find robust solution")
		}

		fmt.Printf("\nRobust solution (knee point) locations: ")
		for _, loc := range robustSolution.Locations {
			fmt.Printf("%d ", loc)
		}
		fmt.Printf("\nObjective values (Huff, PartiallyBinary, ParetoHuff): %.6f%%, %.6f%%, %.6f%%\n",
			robustSolution.Objectives[0],
			robustSolution.Objectives[1],
			robustSolution.Objectives[2])
	}
}
