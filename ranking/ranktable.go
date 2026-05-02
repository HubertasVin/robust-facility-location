package ranking

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
)

// RankTable stores rank scores for each candidate location.
// Ranks persist across problem instances to transfer learned experience.
type RankTable struct {
	ranks map[int]float64
}

// NewRankTable creates an empty RankTable.
func NewRankTable() *RankTable {
	return &RankTable{ranks: make(map[int]float64)}
}

// Get returns the rank score for a location, defaulting to 0.
func (rt *RankTable) Get(loc int) float64 {
	return rt.ranks[loc]
}

// Update adjusts the rank for a location by delta.
func (rt *RankTable) Update(loc int, delta float64) {
	rt.ranks[loc] += delta
}

// Len returns the number of entries.
func (rt *RankTable) Len() int {
	return len(rt.ranks)
}

// MinMax returns the minimum and maximum rank values.
func (rt *RankTable) MinMax() (min, max float64) {
	first := true
	for _, v := range rt.ranks {
		if first {
			min, max = v, v
			first = false
		} else {
			if v < min {
				min = v
			}
			if v > max {
				max = v
			}
		}
	}
	return
}

// Save writes the RankTable to a file.
func (rt *RankTable) Save(filename string) error {
	f, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("saving RankTable: %w", err)
	}
	defer f.Close()

	w := bufio.NewWriter(f)
	for loc, rank := range rt.ranks {
		fmt.Fprintf(w, "%d %f\n", loc, rank)
	}
	if err := w.Flush(); err != nil {
		return err
	}
	fmt.Printf("RankTable saved to %s\n", filename)
	return nil
}

// Load reads a RankTable from a file.
func (rt *RankTable) Load(filename string) error {
	f, err := os.Open(filename)
	if err != nil {
		return fmt.Errorf("loading RankTable: %w", err)
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		parts := strings.Fields(scanner.Text())
		if len(parts) < 2 {
			continue
		}
		loc, err := strconv.Atoi(parts[0])
		if err != nil {
			continue
		}
		rank, err := strconv.ParseFloat(parts[1], 64)
		if err != nil {
			continue
		}
		rt.ranks[loc] = rank
	}
	fmt.Printf("RankTable loaded from %s (%d entries)\n", filename, len(rt.ranks))
	return scanner.Err()
}

// Initialize sets all candidates in the list to initial rank 0 if not present.
func (rt *RankTable) Initialize(candidates []int) {
	for _, loc := range candidates {
		if _, exists := rt.ranks[loc]; !exists {
			rt.ranks[loc] = 0.0
		}
	}
}
