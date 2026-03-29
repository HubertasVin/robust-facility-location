package problem

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
)

// DemandPoint represents a demand point with latitude, longitude, and weight.
type DemandPoint struct {
	Lat    float64
	Lon    float64
	Weight float64
}

// Problem holds all data for a facility location problem instance.
type Problem struct {
	// Demand points
	Demands []DemandPoint

	// Pre-existing facilities: indices and qualities
	J  []int
	QJ []int

	// Candidate locations: indices and qualities
	L  []int
	QL []int

	// Lower-triangular distance matrix (Haversine)
	DM [][]float64
}

// LoadProblem reads the problem file and demands file, builds the distance matrix,
// and returns a fully initialized Problem.
func LoadProblem(problemFile, demandsFile string) (*Problem, error) {
	p := &Problem{}

	if err := p.loadFacilities(problemFile); err != nil {
		return nil, fmt.Errorf("loading facilities: %w", err)
	}
	if err := p.loadDemands(demandsFile); err != nil {
		return nil, fmt.Errorf("loading demands: %w", err)
	}
	p.buildDistanceMatrix()

	return p, nil
}

func (p *Problem) loadFacilities(filename string) error {
	f, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	scanner.Split(bufio.ScanWords)

	// Skip until "-----BEGIN-----"
	for scanner.Scan() {
		if scanner.Text() == "-----BEGIN-----" {
			break
		}
	}

	readInt := func() (int, error) {
		if !scanner.Scan() {
			return 0, fmt.Errorf("unexpected end of file")
		}
		return strconv.Atoi(strings.TrimSpace(scanner.Text()))
	}

	// Pre-existing facilities
	n, err := readInt()
	if err != nil {
		return fmt.Errorf("reading J count: %w", err)
	}
	p.J = make([]int, n)
	p.QJ = make([]int, n)
	for i := 0; i < n; i++ {
		if p.J[i], err = readInt(); err != nil {
			return err
		}
		if p.QJ[i], err = readInt(); err != nil {
			return err
		}
	}

	// Candidate locations
	n, err = readInt()
	if err != nil {
		return fmt.Errorf("reading L count: %w", err)
	}
	p.L = make([]int, n)
	p.QL = make([]int, n)
	for i := 0; i < n; i++ {
		if p.L[i], err = readInt(); err != nil {
			return err
		}
		if p.QL[i], err = readInt(); err != nil {
			return err
		}
	}

	return scanner.Err()
}

func (p *Problem) loadDemands(filename string) error {
	f, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	scanner.Split(bufio.ScanWords)

	readInt := func() (int, error) {
		if !scanner.Scan() {
			return 0, fmt.Errorf("unexpected end of file")
		}
		return strconv.Atoi(strings.TrimSpace(scanner.Text()))
	}

	readFloat := func() (float64, error) {
		if !scanner.Scan() {
			return 0, fmt.Errorf("unexpected end of file")
		}
		return strconv.ParseFloat(strings.TrimSpace(scanner.Text()), 64)
	}

	n, err := readInt()
	if err != nil {
		return fmt.Errorf("reading demand count: %w", err)
	}

	p.Demands = make([]DemandPoint, n)
	for i := range n {
		if p.Demands[i].Lat, err = readFloat(); err != nil {
			return err
		}
		if p.Demands[i].Lon, err = readFloat(); err != nil {
			return err
		}
		if p.Demands[i].Weight, err = readFloat(); err != nil {
			return err
		}
	}

	return scanner.Err()
}
