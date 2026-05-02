package report

import (
	"fmt"
	"reflect"
	"strconv"
	"strings"
	"sync"

	"github.com/HubertasVin/robust-facility-location/problem"
	"github.com/HubertasVin/robust-facility-location/ranking"
)

type SolutionsTableLogger struct {
	table          *SolutionsTable
	behaviorCols   []string
	behaviorColSet map[string]struct{}

	mu   sync.Mutex
	seen map[string]struct{}
}

func behaviorName(b problem.CustomerBehaviorModel) string {
	if b == nil {
		return "<nil>"
	}
	t := reflect.TypeOf(b)
	if t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	if name := t.Name(); name != "" {
		return name
	}
	return fmt.Sprintf("%T", b)
}

func NewSolutionsTableLogger(filename string, behaviors []problem.CustomerBehaviorModel) (*SolutionsTableLogger, error) {
	cols := make([]string, 0, len(behaviors))
	colSet := make(map[string]struct{}, len(behaviors))
	for _, b := range behaviors {
		name := behaviorName(b)
		cols = append(cols, name)
		colSet[name] = struct{}{}
	}

	t, err := NewSolutionsTable(filename, cols)
	if err != nil {
		return nil, err
	}

	return &SolutionsTableLogger{
		table:          t,
		behaviorCols:   cols,
		behaviorColSet: colSet,
		seen:           make(map[string]struct{}, 1024),
	}, nil
}

func rowKey(locations []int, alignedObjectives []float64) string {
	// Key ignores stage/iter so we de-dup the same solution even if re-evaluated.
	var b strings.Builder
	for i, loc := range locations {
		if i > 0 {
			b.WriteByte(',')
		}
		b.WriteString(strconv.Itoa(loc))
	}
	b.WriteByte('|')
	for i, v := range alignedObjectives {
		if i > 0 {
			b.WriteByte(',')
		}
		// Match the table's 6-decimal formatting.
		b.WriteString(strconv.FormatFloat(v, 'f', 6, 64))
	}
	return b.String()
}

func (l *SolutionsTableLogger) recordIfNew(stage string, iter int, locations []int, alignedObjectives []float64) error {
	key := rowKey(locations, alignedObjectives)
	l.mu.Lock()
	if _, ok := l.seen[key]; ok {
		l.mu.Unlock()
		return nil
	}
	l.seen[key] = struct{}{}
	l.mu.Unlock()
	return l.table.Record(stage, iter, locations, alignedObjectives)
}

// Record implements ranking.EvaluationLogger.
func (l *SolutionsTableLogger) Record(stage string, iter int, locations []int, behaviors []problem.CustomerBehaviorModel, objectives []float64) error {
	// Fast path: objectives already aligned to our columns.
	if len(behaviors) == len(l.behaviorCols) && len(objectives) == len(l.behaviorCols) {
		aligned := true
		for i := range behaviors {
			if behaviorName(behaviors[i]) != l.behaviorCols[i] {
				aligned = false
				break
			}
		}
		if aligned {
			return l.recordIfNew(stage, iter, locations, objectives)
		}
	}

	// Map by behavior name, then re-align into our fixed columns.
	byName := make(map[string]float64, len(objectives))
	for i, b := range behaviors {
		if i >= len(objectives) {
			break
		}
		byName[behaviorName(b)] = objectives[i]
	}

	alignedObjectives := make([]float64, len(l.behaviorCols))
	for i, col := range l.behaviorCols {
		if v, ok := byName[col]; ok {
			alignedObjectives[i] = v
			continue
		}
		// If a value is missing, leave as 0.0 (still keeps rectangular TSV).
	}

	return l.recordIfNew(stage, iter, locations, alignedObjectives)
}

func (l *SolutionsTableLogger) Close() error {
	return l.table.Close()
}

var _ ranking.EvaluationLogger = (*SolutionsTableLogger)(nil)
