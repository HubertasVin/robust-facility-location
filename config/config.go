package config

import (
	"os"
	"strconv"
)

// Config holds all configurable parameters for the RL facility location solver.
type Config struct {
	// Problem parameters
	ProblemFile   string
	DemandsFile   string
	MaxFacilities int

	// Q-Learning parameters
	TrainingEpisodes int
	Alpha            float64 // Learning rate
	Gamma            float64 // Discount factor
	FixedEpsilon     bool    // Use fixed epsilon if true; slope function if false
	Epsilon          float64 // Exploration rate
	QTableFile       string
	NumTrainingRuns  int
	WindowSize       int
	EpsilonSlope     float64 // Computed from TrainingEpisodes
	UseFinalRewardQ  bool    // Use final reward to update Q-values
	PerformTraining  bool    // If true, train; else greedy from loaded Q-table
}

// Load reads configuration from environment variables, falling back to defaults.
func Load() *Config {
	episodes := envInt("RL_EPISODES", 20000)

	return &Config{
		ProblemFile:      envString("RL_PROBLEM_FILE", "CFLP.dat"),
		DemandsFile:      envString("RL_DEMANDS_FILE", "demands.dat"),
		MaxFacilities:    envInt("RL_MAX_FACILITIES", 3),
		TrainingEpisodes: episodes,
		Alpha:            envFloat("RL_ALPHA", 0.8),
		Gamma:            envFloat("RL_GAMMA", 0.0),
		FixedEpsilon:     envBool("RL_FIXED_EPSILON", true),
		Epsilon:          envFloat("RL_EPSILON", 0.2),
		QTableFile:       envString("RL_QTABLE_FILE", "qtable.dat"),
		NumTrainingRuns:  envInt("RL_NUM_RUNS", 1),
		WindowSize:       envInt("RL_WINDOW_SIZE", 1),
		EpsilonSlope:     (float64(episodes) / 2.0) / 9.0,
		UseFinalRewardQ:  envBool("RL_FINAL_REWARD_Q", false),
		PerformTraining:  envBool("RL_TRAINING_MODE", true),
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
