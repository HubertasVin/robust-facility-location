package ranking

import (
	"math/rand/v2"
	"sort"
)

// Individual represents a solution in the population.
type Individual struct {
	Locations []int   // Facility location IDs (elements of Problem.L)
	Utility   float64 // Evaluated utility of this solution
}

// Copy creates a deep copy of the individual.
func (ind *Individual) Copy() *Individual {
	locs := make([]int, len(ind.Locations))
	copy(locs, ind.Locations)
	return &Individual{
		Locations: locs,
		Utility:   ind.Utility,
	}
}

// Population manages a collection of best solutions found.
type Population struct {
	individuals []*Individual
	maxSize     int
}

// NewPopulation creates an empty population with the given maximum size.
func NewPopulation(maxSize int) *Population {
	return &Population{
		individuals: make([]*Individual, 0, maxSize),
		maxSize:     maxSize,
	}
}

// Len returns the current population size.
func (p *Population) Len() int {
	return len(p.individuals)
}

// Add attempts to add a new individual to the population.
// Returns true if the individual was added.
func (p *Population) Add(ind *Individual) bool {
	// If population is not full, always add
	if len(p.individuals) < p.maxSize {
		p.individuals = append(p.individuals, ind)
		p.sortByUtility()
		return true
	}

	// If full, only add if better than worst
	worst := p.individuals[len(p.individuals)-1]
	if ind.Utility > worst.Utility {
		p.individuals[len(p.individuals)-1] = ind
		p.sortByUtility()
		return true
	}

	return false
}

// sortByUtility sorts individuals by utility in descending order.
func (p *Population) sortByUtility() {
	sort.Slice(p.individuals, func(i, j int) bool {
		return p.individuals[i].Utility > p.individuals[j].Utility
	})
}

// Best returns the best individual in the population, or nil if empty.
func (p *Population) Best() *Individual {
	if len(p.individuals) == 0 {
		return nil
	}
	return p.individuals[0]
}

// RandomSelect returns a random individual from the population.
func (p *Population) RandomSelect() *Individual {
	if len(p.individuals) == 0 {
		return nil
	}
	return p.individuals[rand.IntN(len(p.individuals))]
}

// AverageUtility returns the average utility of the population.
func (p *Population) AverageUtility() float64 {
	if len(p.individuals) == 0 {
		return 0
	}
	sum := 0.0
	for _, ind := range p.individuals {
		sum += ind.Utility
	}
	return sum / float64(len(p.individuals))
}
