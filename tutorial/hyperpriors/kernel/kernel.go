package kernel

import (
	"bitbucket.org/dtolpin/gogp/kernel"
)

// The similarity kernel.
type simil struct{}

var Simil simil

func (simil) Observe(x []float64) float64 {
	const (
		c1 = iota // trend scale
		c2        // season scale
		l1        // trend length scale
		l2        // season length scale
		p         // season period
		xa        // first point
		xb        // second point
	)

	return x[c1]*kernel.Matern52.Cov(x[l1], x[xa], x[xb]) +
		x[c2]*kernel.Periodic.Cov(x[l2], x[p], x[xa], x[xb])
}

func (simil) NTheta() int { return 5 }

// The noise kernel.
type noise struct{}

var Noise noise

func (n noise) Observe(x []float64) float64 {
	return kernel.UniformNoise.Observe(x)
}

func (noise) NTheta() int { return 1 }
