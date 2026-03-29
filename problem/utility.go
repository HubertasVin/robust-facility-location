package problem

// UtilityBinary computes the binary customer-choice utility for a set of
// candidate facility indices X (indices into L/QL).
func (p *Problem) UtilityBinary(X []int) float64 {
	var total, utility float64

	for i, dp := range p.Demands {
		total += dp.Weight

		// Best attractiveness from pre-existing facilities
		bestJ := -1.0
		for jIdx, jLoc := range p.J {
			attr := float64(p.QJ[jIdx]) / (1.0 + p.Distance(i, jLoc))
			if attr > bestJ {
				bestJ = attr
			}
		}

		// Best attractiveness from selected candidate facilities
		bestX := -1.0
		for _, xIdx := range X {
			attr := float64(p.QL[xIdx]) / (1.0 + p.Distance(i, p.L[xIdx]))
			if attr > bestX {
				bestX = attr
			}
		}

		if bestX > bestJ {
			utility += dp.Weight
		} else if bestX == bestJ {
			utility += dp.Weight / 2.0
		}
	}

	if total == 0 {
		return 0
	}
	return utility / total * 100.0
}

// UtilityProportional computes the proportional customer-choice utility
// for a set of candidate facility indices X.
func (p *Problem) UtilityProportional(X []int) float64 {
	var total, utility float64

	for i, dp := range p.Demands {
		total += dp.Weight

		var attrJ float64
		for jIdx, jLoc := range p.J {
			attrJ += float64(p.QJ[jIdx]) / (1.0 + p.Distance(i, jLoc))
		}

		var attrX float64
		for _, xIdx := range X {
			attrX += float64(p.QL[xIdx]) / (1.0 + p.Distance(i, p.L[xIdx]))
		}

		if attrJ+attrX > 0 {
			utility += dp.Weight * attrX / (attrJ + attrX)
		}
	}

	if total == 0 {
		return 0
	}
	return utility / total * 100.0
}
