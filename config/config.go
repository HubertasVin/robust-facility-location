package config

import (
	"os"
	"strconv"

	"github.com/joho/godotenv"
)

func init() {
	_ = godotenv.Load()
}

type Config struct {
	ProblemFile   string
	DemandsFile   string
	MaxFacilities int

	PopulationSize  int
	Iterations      int
	Epsilon         float64
	Alpha           float64
	RankFile        string
	PerformTraining bool

	CheckedSolutionsFile string
	JSONMode             bool
}

func Load() *Config {
	return &Config{
		ProblemFile:          envString("PROBLEM_FILE", "CFLP.dat"),
		DemandsFile:          envString("DEMANDS_FILE", "demands.dat"),
		MaxFacilities:        envInt("MAX_FACILITIES", 3),
		PopulationSize:       envInt("POPULATION_SIZE", 10),
		Iterations:           envInt("ITERATIONS", 10000),
		Epsilon:              envFloat("EPSILON", 0.3),
		Alpha:                envFloat("ALPHA", 0.1),
		RankFile:             envString("RANK_FILE", "ranks.dat"),
		PerformTraining:      envBool("TRAINING_MODE", true),
		CheckedSolutionsFile: envString("CHECKED_SOLUTIONS_FILE", "checked_solutions.tsv"),
		JSONMode:             envBool("JSON_MODE", false),
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
