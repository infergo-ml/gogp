package kernel

import (
	"bitbucket.org/dtolpin/gogp/kernel"
)

// The similarity kernel, just a scaled RBF.
// To add output scale scaling, all one needs to do
// is to multiple the finction value by another parameter.
type simil struct{}

var Simil simil

func (simil) Observe(x []float64) float64 {
	return x[0] * kernel.Normal.Observe(x[1:])
}

func (simil) NTheta() int { return 2 }

// The noise kernel, uniform noise scaled by a likely value.
// Scaling is tantamount to specifying an initial
// point in the proximity of a reasonable noise variance but
// makes the code easier to write and read.

type Noise float64

func (n Noise) Observe(x []float64) float64 {
	return float64(n) * kernel.UniformNoise.Observe(x)
}

func (Noise) NTheta() int { return 1 }
