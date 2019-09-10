package kernel

import (
	"math"
)

// Noise kernels

// ConstantNoise is a noise kernel assigning the same fixed
// noise to all points. Used as a default when no noise kernel
// is given.
type ConstantNoise float64

func (nk ConstantNoise) Observe([]float64) float64 {
	return float64(nk)
}

// UniformNoise is a noise kernel for learning the same noise
// variance for all points.
type UniformNoise struct{}

func (nk UniformNoise) Observe(x []float64) float64 {
	return math.Exp(x[0])
}