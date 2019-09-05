package kernel

import (
	"math"
)

// Noise kernels

// ConstantNoise is a kernel given the same fixed noise for all
// points. Used as a default when no noise kernel is given.
type ConstantNoise float64
func (nk ConstantNoise) Observe([]float64) float64 {
	return float64(nk)
}

// UniformNoise is a kernel for learning the same noise variance
// for all points.
type UniformNoise struct {}
func (nk UniformNoise) Observe (x []float64) float64 {
	return math.Exp(x[0])
}
