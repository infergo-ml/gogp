// Package kernel provides a library of primitive kernels.
package kernel

import (
	"math"
)

// Type normal is the normal kernel type. A normal kernel
// has a single parameter, the length scale.
type normal struct{}

// Singleton for the normal kernel
var Normal normal

func (k normal) Observe(x []float64) float64 {
	return k.Cov(x[0], x[1], x[2])
}

func (k normal) Cov(l, xa, xb float64) float64 {
	d := (xa - xb) / l
	return math.Exp(-d * d / 2)
}

// Type periodic is the exponential periodic kernel type. An
// exponential periodic kernel has two parameters, the length
// scale and the period.
type periodic struct{}

// Singleton for the periodic kernel
var Periodic periodic

func (k periodic) Observe(x []float64) float64 {
	return k.Cov(x[0], x[1], x[2], x[3])
}

func (k periodic) Cov(l, p, xa, xb float64) float64 {
	d := math.Sin(math.Pi*math.Abs(xa-xb)/p) / l
	return math.Exp(-2 * d * d)
}

// The constants below are for Matern kernels.
const (
	sqrt3 = 1.7320508075688772
	sqrt5 = 2.2360679774997900
)

// Type matern32 is the Matern(nu=3/2) kernel type. A Matern
// kernel has a single parameter, the length scale.
type matern32 struct{}

// Singleton for Matern32
var Matern32 matern32

func (k matern32) Observe(x []float64) float64 {
	return k.Cov(x[0], x[1], x[2])
}

func (k matern32) Cov(l, xa, xb float64) float64 {
	d := math.Abs(xa-xb) / l
	return (1 + sqrt3*d) * math.Exp(-sqrt3*d)
}

// Type matern52 is the Matern(nu=5/2) kernel type. A Matern
// kernel has a single parameter, the length scale.
type matern52 struct{}

var Matern52 matern52

func (k matern52) Observe(x []float64) float64 {
	return k.Cov(x[0], x[1], x[2])
}

func (k matern52) Cov(l, xa, xb float64) float64 {
	d := math.Abs(xa-xb) / l
	return (1 + sqrt5*d + 5/3*d*d) * math.Exp(-sqrt5*d)
}
