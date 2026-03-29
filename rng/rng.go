package rng

import (
	"math/rand"
)

var src *rand.Rand

// Seed initializes the global RNG with the given seed.
func Seed(seed uint64) {
	src = rand.New(rand.NewSource(int64(seed)))
}

// Float64 returns a random float64 in [0.0, 1.0).
func Float64() float64 {
	return src.Float64()
}

// Intn returns a random int in [0, n).
func Intn(n int) int {
	return src.Intn(n)
}
