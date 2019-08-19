// Package kernel provides a library of primitive kernels.
package kernel

import (
	"math"
)

// Type Gaussian is the Gaussian kernel type. A Gaussian kernel
// has a single parameter, the length scale.
type Gaussian struct{}

func (k Gaussian) Observe(x []float64) float64 {
	l := math.Exp(x[0])
	d := (x[1] - x[2]) / l
	return math.Exp(-d * d / 2)
}

type Periodic struct{}

// Type Periodic is the exponential periodic kernel type. An
// exponential periodic kernel has two parameters, the length
// scale and the period.
func (k Periodic) Observe(x []float64) float64 {
	l := math.Exp(x[0])
	p := math.Exp(x[1])
	d := math.Sin(math.Pi*math.Abs(x[2]-x[3])/p) / l
	return math.Exp(-2 * d * d)
}

const (
	sqrt3 = 1.7320508075688772
	sqrt5 = 2.2360679774997900
)

// Type Matern32 is the Matern(nu=3/2) kernel type. A Matern
// kernel has a single parameter, the length scale.
type Matern32 struct{}

func (k Matern32) Observe(x []float64) float64 {
	l := math.Exp(x[0])
	d := math.Abs(x[1]-x[2]) / l
	return (1 + sqrt3*d) * math.Exp(-sqrt3*d)
}

// Type Matern52 is the Matern(nu=5/2) kernel type. A Matern
// kernel has a single parameter, the length scale.
func (k Matern52) Observe(x []float64) float64 {
	l := math.Exp(x[0])
	d := math.Abs(x[1]-x[2]) / l
	return (1 + sqrt5*d + 5/3*d*d) * math.Exp(-sqrt5*d)
}
