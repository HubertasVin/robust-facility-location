package problem

// CustomerBehaviorModel computes utility (market share in %) of candidate
// facilities X for one customer-choice rule.
type CustomerBehaviorModel interface {
	Utility(p *Problem, X []int) float64
}

// UtilityFunc allows passing a function as a customer behavior model.
type UtilityFunc func(p *Problem, X []int) float64

// Utility implements CustomerBehaviorModel.
func (f UtilityFunc) Utility(p *Problem, X []int) float64 {
	return f(p, X)
}

// BinaryModel is the winner-takes-most customer behavior.
type BinaryModel struct{}

// HuffModel is the proportional customer behavior model.
type HuffModel struct{}

// PartiallyBinaryModel picks one best facility per firm, then splits demand
// proportionally among those winners.
type PartiallyBinaryModel struct{}

// ParetoHuffModel ignores facilities dominated in both distance and quality,
// then applies Huff proportional allocation on the Pareto set.
type ParetoHuffModel struct{}

func attractiveness(distance float64, quality int) float64 {
	return float64(quality) / (distance + 1.0)
}

func toPercent(utility, total float64) float64 {
	if total == 0 {
		return 0
	}
	return utility / total * 100.0
}

// Utility computes binary customer-choice utility.
func (BinaryModel) Utility(p *Problem, X []int) float64 {
	var total, utility float64

	for i, dp := range p.Demands {
		total += dp.Weight

		bestJ := -1.0
		for jIdx, jLoc := range p.J {
			attr := attractiveness(p.Distance(i, jLoc), p.QJ[jIdx])
			if attr > bestJ {
				bestJ = attr
			}
		}

		bestX := -1.0
		for _, xIdx := range X {
			attr := attractiveness(p.Distance(i, p.L[xIdx]), p.QL[xIdx])
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

	return toPercent(utility, total)
}

// Utility computes Huff (proportional) customer-choice utility.
func (HuffModel) Utility(p *Problem, X []int) float64 {
	var total, utility float64

	for i, dp := range p.Demands {
		total += dp.Weight

		var attrJ float64
		for jIdx, jLoc := range p.J {
			attrJ += attractiveness(p.Distance(i, jLoc), p.QJ[jIdx])
		}

		var attrX float64
		for _, xIdx := range X {
			attrX += attractiveness(p.Distance(i, p.L[xIdx]), p.QL[xIdx])
		}

		denom := attrJ + attrX
		if denom > 0 {
			utility += dp.Weight * attrX / denom
		}
	}

	return toPercent(utility, total)
}

// Utility computes the partially binary customer-choice utility.
func (PartiallyBinaryModel) Utility(p *Problem, X []int) float64 {
	var total, utility float64

	for i, dp := range p.Demands {
		total += dp.Weight

		bestJ := 0.0
		for jIdx, jLoc := range p.J {
			attr := attractiveness(p.Distance(i, jLoc), p.QJ[jIdx])
			if attr > bestJ {
				bestJ = attr
			}
		}

		bestX := 0.0
		for _, xIdx := range X {
			attr := attractiveness(p.Distance(i, p.L[xIdx]), p.QL[xIdx])
			if attr > bestX {
				bestX = attr
			}
		}

		denom := bestJ + bestX
		if denom > 0 {
			utility += dp.Weight * bestX / denom
		}
	}

	return toPercent(utility, total)
}

// Utility computes Pareto-Huff customer-choice utility.
func (ParetoHuffModel) Utility(p *Problem, X []int) float64 {
	type facility struct {
		distance float64
		quality  int
		attr     float64
		ours     bool
	}

	var total, utility float64

	for i, dp := range p.Demands {
		total += dp.Weight

		facilities := make([]facility, 0, len(p.J)+len(X))
		for jIdx, jLoc := range p.J {
			d := p.Distance(i, jLoc)
			facilities = append(facilities, facility{
				distance: d,
				quality:  p.QJ[jIdx],
				attr:     attractiveness(d, p.QJ[jIdx]),
				ours:     false,
			})
		}
		for _, xIdx := range X {
			d := p.Distance(i, p.L[xIdx])
			facilities = append(facilities, facility{
				distance: d,
				quality:  p.QL[xIdx],
				attr:     attractiveness(d, p.QL[xIdx]),
				ours:     true,
			})
		}

		paretoMask := make([]bool, len(facilities))
		for a := range facilities {
			paretoMask[a] = true
		}

		for a := range facilities {
			for b := range facilities {
				if a == b {
					continue
				}
				dominated := facilities[b].distance <= facilities[a].distance &&
					facilities[b].quality >= facilities[a].quality &&
					(facilities[b].distance < facilities[a].distance ||
						facilities[b].quality > facilities[a].quality)
				if dominated {
					paretoMask[a] = false
					break
				}
			}
		}

		var totalAttr, ourAttr float64
		for idx, keep := range paretoMask {
			if !keep {
				continue
			}
			totalAttr += facilities[idx].attr
			if facilities[idx].ours {
				ourAttr += facilities[idx].attr
			}
		}

		if totalAttr > 0 {
			utility += dp.Weight * ourAttr / totalAttr
		}
	}

	return toPercent(utility, total)
}
