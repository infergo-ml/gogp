package kernel

import (
	"bitbucket.org/dtolpin/gogp/kernel"
)

// The similarity kernel.
type simil struct{}

var Simil simil

func (simil) Observe(x []float64) float64 {
	c1 := x[0] // trend output scale
	c2 := x[1] // season output scale
	l1 := x[2] // trend length scale
	l2 := x[3] // season length scale
	p := x[4] // season period
	xa := x[5] // first point
	xb := x[6] // second point

	return c1 * kernel.Matern52.Cov(l1, xa, xb) +
		   c2 * kernel.Periodic.Cov(l2, p, xa, xb)
}

func (simil) NTheta() int { return 5 }

// The noise kernel.
type Noise float64

func (n Noise) Observe(x []float64) float64 {
	return float64(n) * kernel.UniformNoise.Observe(x)
}

func (Noise) NTheta() int { return 1 }
