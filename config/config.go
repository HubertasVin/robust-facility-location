package config

import (
	"os"
	"strconv"

	"github.com/joho/godotenv"
)

func init() {
	// Load .env file if it exists (silently ignore if not found)
	_ = godotenv.Load()
}

// Config holds all configurable parameters for the FLARC/PL facility location solver.
type Config struct {
	// Problem parameters
	ProblemFile   string
	DemandsFile   string
	MaxFacilities int

	// FLARC/PL parameters
	PopulationSize  int     // Maximum population size (n_P)
	Iterations      int     // Number of iterations
	Epsilon         float64 // Mutation probability
	Alpha           float64 // Learning rate for rank updates
	RankFile        string  // File to save/load rank scores
	PerformTraining bool    // If true, run optimization; else use stored ranks

	// Output / reporting
	CheckedSolutionsFile string // TSV file with all evaluated solutions and objectives
}

// Load reads configuration from environment variables, falling back to defaults.
func Load() *Config {
	return &Config{
		ProblemFile:          envString("RL_PROBLEM_FILE", "CFLP.dat"),
		DemandsFile:          envString("RL_DEMANDS_FILE", "demands.dat"),
		MaxFacilities:        envInt("RL_MAX_FACILITIES", 3),
		PopulationSize:       envInt("RL_POPULATION_SIZE", 10),
		Iterations:           envInt("RL_ITERATIONS", 10000),
		Epsilon:              envFloat("RL_EPSILON", 0.3),
		Alpha:                envFloat("RL_ALPHA", 0.1),
		RankFile:             envString("RL_RANK_FILE", "ranks.dat"),
		PerformTraining:      envBool("RL_TRAINING_MODE", true),
		CheckedSolutionsFile: envString("RL_CHECKED_SOLUTIONS_FILE", "checked_solutions.tsv"),
	}
}

func envString(key, defaultVal string) string {
	if val, ok := os.LookupEnv(key); ok {
		return val
	}
	return defaultVal
}

func envInt(key string, defaultVal int) int {
	if val, ok := os.LookupEnv(key); ok {
		if i, err := strconv.Atoi(val); err == nil {
			return i
		}
	}
	return defaultVal
}

func envFloat(key string, defaultVal float64) float64 {
	if val, ok := os.LookupEnv(key); ok {
		if f, err := strconv.ParseFloat(val, 64); err == nil {
			return f
		}
	}
	return defaultVal
}

func envBool(key string, defaultVal bool) bool {
	if val, ok := os.LookupEnv(key); ok {
		return val == "true" || val == "1" || val == "TRUE"
	}
	return defaultVal
}
