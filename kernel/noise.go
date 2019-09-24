package kernel

import (
	"math"
)

// Noise kernels

// ConstantNoise is a noise kernel assigning the same fixed
// noise to all points. Used as a default when no noise kernel
// is given.
type ConstantNoise float64

func (nk ConstantNoise) Observe(x []float64) float64 {
	return float64(nk)
}

func (ConstantNoise) NTheta() int {
	return 0
}

// UniformNoise is a noise kernel for learning the same noise
// variance for all points.
type uniformNoise struct{}

var UniformNoise uniformNoise

func (nk uniformNoise) Observe(x []float64) float64 {
	return math.Exp(x[0])
}

func (uniformNoise) NTheta() int {
	return 1
}
