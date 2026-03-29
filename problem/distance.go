package problem

import (
	"math"
)

const degToRad = math.Pi / 180.0

// HaversineDistance computes the great-circle distance in km between two
// geographic points specified in degrees.
func HaversineDistance(lat1, lon1, lat2, lon2 float64) float64 {
	dlon := math.Abs(lon1 - lon2)
	dlat := math.Abs(lat1 - lat2)

	a := math.Pow(math.Sin(dlat/2*degToRad), 2) +
		math.Cos(lat1*degToRad)*math.Cos(lat2*degToRad)*
			math.Pow(math.Sin(dlon/2*degToRad), 2)
	c := 2 * math.Atan2(math.Sqrt(a), math.Sqrt(1-a))
	d := 6371.0 * c

	return math.Round(d)
}

// buildDistanceMatrix constructs a lower-triangular distance matrix
// between all demand points.
func (p *Problem) buildDistanceMatrix() {
	n := len(p.Demands)
	p.DM = make([][]float64, n)
	for i := 0; i < n; i++ {
		p.DM[i] = make([]float64, i+1)
		for j := 0; j <= i; j++ {
			p.DM[i][j] = HaversineDistance(
				p.Demands[i].Lat, p.Demands[i].Lon,
				p.Demands[j].Lat, p.Demands[j].Lon,
			)
		}
	}
}

// Distance returns the precomputed distance between demand points i and j.
func (p *Problem) Distance(i, j int) float64 {
	if i >= j {
		return p.DM[i][j]
	}
	return p.DM[j][i]
}
