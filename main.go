package main

import (
	"encoding/json"
	"fmt"
	"log"
	"os"

	"github.com/HubertasVin/robust-facility-location/config"
	"github.com/HubertasVin/robust-facility-location/problem"
	"github.com/HubertasVin/robust-facility-location/ranking"
	"github.com/HubertasVin/robust-facility-location/report"
	"github.com/HubertasVin/robust-facility-location/solution"
)

type JSONResult struct {
	MaxFacilities int                  `json:"max_facilities"`
	Iterations    int                  `json:"iterations"`
	KneePoint     *solution.Solution   `json:"knee_point"`
	ParetoFront   []*solution.Solution `json:"pareto_front"`
}

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

	agent := ranking.NewAgent(cfg, prob, behaviorModel)

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

	_ = agent.RankTable.Load(cfg.RankFile)

	if cfg.PerformTraining {
		fmt.Fprintln(os.Stderr, "=== Running FLARC/PL Optimization ===")
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
		fmt.Fprintln(os.Stderr, "=== Finding Robust Solution (Knee Point) ===")
		robustSolution := agent.FindRobustSolution(behaviors)
		if robustSolution == nil {
			log.Fatalf("Failed to find robust solution")
		}

		if cfg.JSONMode {
			outputJSON(cfg, robustSolution, agent.ParetoFront)
		} else {
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
}

func outputJSON(cfg *config.Config, kneePoint *solution.Solution, pf *solution.ParetoFront) {
	result := JSONResult{
		MaxFacilities: cfg.MaxFacilities,
		Iterations:    cfg.Iterations,
		KneePoint:     kneePoint,
	}

	if pf != nil {
		result.ParetoFront = pf.Solutions
	}

	encoder := json.NewEncoder(os.Stdout)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(result); err != nil {
		log.Fatalf("Failed to encode JSON: %v", err)
	}
}