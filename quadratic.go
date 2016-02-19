package ml

import (
	"fmt"
	"math"
	"sync"
)

type Quadratic struct {
	X, Y    []float64
	A, B, C float64
}

type resource struct {
	prop  string
	value float64
}

/*
source: https://www.easycalculation.com/statistics/learn-quadratic-regression.php
Quadratic Regression Equation(y) = a x^2 + b x + c
a = { [ . x2 y * . xx ] - [. xy * . xx2 ] } / { [ . xx * . x2x 2] - [. xx2 ]2 }
b = { [ . xy * . x2x2 ] - [. x2y * . xx2 ] } / { [ . xx * . x2x 2] - [. xx2 ]2 }
c = [ . y / n ] - { b * [ . x / n ] } - { a * [ . x 2 / n ] }
*/
func (q *Quadratic) LoadModel() {
	resources := make(map[string]float64)
	q.calcFirstValues(resources)
	q.calcSecondValues(resources)

	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		q.A = (resources["x^2y"]*resources["xx"] - resources["xy"]*resources["xx^2"]) / (resources["xx"]*resources["x^2x^2"] - math.Pow(resources["xx^2"], 2))
		wg.Done()
	}()

	go func() {
		q.B = (resources["xy"]*resources["x^2x^2"] - resources["x^2y"]*resources["xx^2"]) / (resources["xx"]*resources["x^2x^2"] - math.Pow(resources["xx^2"], 2))
		wg.Done()
	}()

	wg.Wait()
	q.C = (resources["sumy"] / resources["n"]) - (q.B * resources["sumx"] / resources["n"]) - (q.A * resources["sumx^2"] / resources["n"])

	return
}

func (q *Quadratic) calcFirstValues(resources map[string]float64) {
	ch := make(chan resource)
	go func() {
		ch <- resource{prop: "n", value: float64(len(q.X))}
	}()
	go func() {
		ch <- resource{prop: "sumx", value: Sum(q.X)}
	}()
	go func() {
		ch <- resource{prop: "sumy", value: Sum(q.Y)}
	}()
	go func() {
		ch <- resource{prop: "sumx^2", value: SumPow(q.X, 2)}
	}()
	go func() {
		ch <- resource{prop: "sumx^3", value: SumPow(q.X, 3)}
	}()
	go func() {
		ch <- resource{prop: "sumx^4", value: SumPow(q.X, 4)}
	}()
	go func() {
		ch <- resource{prop: "sumx*y", value: SumMult(q.X, q.Y)}
	}()
	go func() {
		ch <- resource{prop: "sumx^2*y", value: SumMult(PowArray(q.X, 2), q.Y)}
	}()

	calcResources := 0
	for r := range ch {
		calcResources++
		resources[r.prop] = r.value
		if calcResources == 8 {
			close(ch)
		}
	}
}

func (q *Quadratic) calcSecondValues(resources map[string]float64) {
	ch := make(chan resource)
	go func() {
		ch <- resource{prop: "xx", value: (resources["sumx^2"] - (math.Pow(resources["sumx"], 2) / resources["n"]))}
	}()
	go func() {
		ch <- resource{prop: "xy", value: resources["sumx*y"] - ((resources["sumx"] * resources["sumy"]) / resources["n"])}
	}()
	go func() {
		ch <- resource{prop: "xx^2", value: resources["sumx^3"] - ((resources["sumx^2"] * resources["sumx"]) / resources["n"])}
	}()
	go func() {
		ch <- resource{prop: "x^2y", value: resources["sumx^2*y"] - ((resources["sumx^2"] * resources["sumy"]) / resources["n"])}
	}()
	go func() {
		ch <- resource{prop: "x^2x^2", value: resources["sumx^4"] - (math.Pow(resources["sumx^2"], 2) / resources["n"])}
	}()

	calcResources := 0
	for r := range ch {
		calcResources++
		resources[r.prop] = r.value
		if calcResources == 5 {
			close(ch)
		}
	}
}

func PowArray(x []float64, exp float64) (sum []float64) {
	sum = make([]float64, len(x), len(x))
	for i, v := range x {
		sum[i] = math.Pow(v, exp)
	}
	return
}

func SumMult(x, y []float64) (sum float64) {
	for i, v := range x {
		sum += v * y[i]
	}
	return

}

func Sum(values []float64) (sum float64) {
	for _, v := range values {
		sum += v
	}
	return
}

func SumPow(values []float64, exp float64) (sum float64) {
	for _, v := range values {
		sum += math.Pow(v, exp)
	}
	return
}

func (q Quadratic) String() string {
	return fmt.Sprintf("{%q:%f,%q:%f,%q:%f}", "a", q.A, "b", q.B, "c", q.C)
}
