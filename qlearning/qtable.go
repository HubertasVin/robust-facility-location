package qlearning

import (
	"bufio"
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"
)

// QTable is a string-keyed value function table.
type QTable struct {
	data map[string]float64
}

// NewQTable creates an empty Q-table.
func NewQTable() *QTable {
	return &QTable{data: make(map[string]float64)}
}

// Get returns the Q-value for a key, defaulting to 0.
func (q *QTable) Get(key string) float64 {
	return q.data[key] // zero-value for missing keys is 0.0
}

// Set sets the Q-value for a key.
func (q *QTable) Set(key string, value float64) {
	q.data[key] = value
}

// Has checks whether a key exists.
func (q *QTable) Has(key string) bool {
	_, ok := q.data[key]
	return ok
}

// Len returns the number of entries.
func (q *QTable) Len() int {
	return len(q.data)
}

// Save writes the Q-table to a file.
func (q *QTable) Save(filename string) error {
	f, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("saving Q-table: %w", err)
	}
	defer f.Close()

	w := bufio.NewWriter(f)
	for key, val := range q.data {
		fmt.Fprintf(w, "%s %f\n", key, val)
	}
	if err := w.Flush(); err != nil {
		return err
	}
	fmt.Printf("Q-Table saved to %s\n", filename)
	return nil
}

// Load reads a Q-table from a file.
func (q *QTable) Load(filename string) error {
	f, err := os.Open(filename)
	if err != nil {
		return fmt.Errorf("loading Q-table: %w", err)
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		parts := strings.Fields(scanner.Text())
		if len(parts) < 2 {
			continue
		}
		val, err := strconv.ParseFloat(parts[1], 64)
		if err != nil {
			continue
		}
		q.data[parts[0]] = val
	}
	fmt.Printf("Q-Table loaded from %s\n", filename)
	return scanner.Err()
}

// CopyFrom replaces this table's contents with those of another.
func (q *QTable) CopyFrom(other *QTable) {
	q.data = make(map[string]float64, len(other.data))
	for k, v := range other.data {
		q.data[k] = v
	}
}

// --- Key helpers ---

// StateActionKey builds the Q-table key for a (state, action) pair.
func StateActionKey(state []int, action int) string {
	var b strings.Builder
	for _, loc := range state {
		b.WriteString(strconv.Itoa(loc))
		b.WriteByte('-')
	}
	b.WriteByte('|')
	b.WriteString(strconv.Itoa(action))
	return b.String()
}

// StateKey builds a string key for a state.
func StateKey(state []int) string {
	var b strings.Builder
	for _, loc := range state {
		b.WriteString(strconv.Itoa(loc))
		b.WriteByte('-')
	}
	return b.String()
}

// CanonicalStateKey returns a sorted state key.
func CanonicalStateKey(state []int) string {
	sorted := make([]int, len(state))
	copy(sorted, state)
	sort.Ints(sorted)
	return StateKey(sorted)
}
