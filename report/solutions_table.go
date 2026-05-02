package report

import (
	"bufio"
	"fmt"
	"os"
	"strconv"
	"strings"
	"sync"
)

// SolutionsTable writes a TSV table of evaluated solutions.
//
// Rows are appended via Record(). This type is safe for use by a single goroutine
// (and also guarded by a mutex in case the caller ever parallelizes evaluations).
type SolutionsTable struct {
	mu           sync.Mutex
	f            *os.File
	w            *bufio.Writer
	behaviorCols []string
	writes       int
}

func NewSolutionsTable(filename string, behaviorColumns []string) (*SolutionsTable, error) {
	f, err := os.Create(filename)
	if err != nil {
		return nil, err
	}

	t := &SolutionsTable{
		f:            f,
		w:            bufio.NewWriterSize(f, 256*1024),
		behaviorCols: append([]string(nil), behaviorColumns...),
	}

	// Header
	cols := []string{"stage", "iter", "locations"}
	cols = append(cols, t.behaviorCols...)
	cols = append(cols, "mean")
	if _, err := t.w.WriteString(strings.Join(cols, "\t") + "\n"); err != nil {
		_ = f.Close()
		return nil, err
	}

	return t, nil
}

// Record appends one row.
//
// - stage: e.g. "train:init", "train:iter", "robust:init", "robust:iter"
// - iter: iteration index, or -1 for initialization
// - locations: facility location IDs
// - objectives: objective values aligned with behaviorColumns
func (t *SolutionsTable) Record(stage string, iter int, locations []int, objectives []float64) error {
	t.mu.Lock()
	defer t.mu.Unlock()

	var b strings.Builder
	b.WriteString(stage)
	b.WriteByte('\t')
	b.WriteString(strconv.Itoa(iter))
	b.WriteByte('\t')

	for i, loc := range locations {
		if i > 0 {
			b.WriteByte(',')
		}
		b.WriteString(strconv.Itoa(loc))
	}

	mean := 0.0
	if len(objectives) > 0 {
		for _, v := range objectives {
			mean += v
		}
		mean /= float64(len(objectives))
	}

	for _, v := range objectives {
		b.WriteByte('\t')
		fmt.Fprintf(&b, "%.6f", v)
	}

	// If objectives were shorter than declared columns, pad to keep TSV rectangular.
	for i := len(objectives); i < len(t.behaviorCols); i++ {
		b.WriteByte('\t')
	}

	b.WriteByte('\t')
	fmt.Fprintf(&b, "%.6f", mean)
	b.WriteByte('\n')

	if _, err := t.w.WriteString(b.String()); err != nil {
		return err
	}

	t.writes++
	if t.writes%1000 == 0 {
		return t.w.Flush()
	}
	return nil
}

func (t *SolutionsTable) Close() error {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.w == nil {
		return nil
	}
	if err := t.w.Flush(); err != nil {
		_ = t.f.Close()
		t.w = nil
		return err
	}
	err := t.f.Close()
	t.w = nil
	return err
}
