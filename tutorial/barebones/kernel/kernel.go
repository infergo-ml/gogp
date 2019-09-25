package kernel

import (
	"bitbucket.org/dtolpin/gogp/kernel"
)

type simil struct{}

var Simil simil

func (simil) Observe(x []float64) float64 {
	return x[0] * kernel.Normal.Observe(x[1:])
}

func (simil) NTheta() int { return 2 }

type Noise float64

func (n Noise) Observe(x []float64) float64 {
	return float64(n) * kernel.UniformNoise.Observe(x)
}

func (Noise) NTheta() int { return 1 }
