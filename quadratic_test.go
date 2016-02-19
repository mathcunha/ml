package ml

import (
	"testing"

	"fmt"
)

func TestQuadractic(t *testing.T) {
	q := Quadratic{X: []float64{2, 4, 6}, Y: []float64{3, 6, 4}}
	q.LoadModel()
	fmt.Println(q)

	q = Quadratic{X: []float64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, Y: []float64{1, 6, 17, 34, 57, 86, 121, 162, 209, 262, 321}}
	q.LoadModel()
	fmt.Println(q)
}
