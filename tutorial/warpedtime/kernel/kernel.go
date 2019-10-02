package kernel

import (
	"bitbucket.org/dtolpin/gogp/kernel"
)

// The similarity kernel.
type simil struct{}

var Simil simil

func (simil) Observe(x []float64) float64 {
	const (
		c = iota // output scale
		l        // length scale
		xa        // first point
		xb        // second point
	)

	return x[c]*kernel.Matern52.Cov(x[l], x[xa], x[xb])
}

func (simil) NTheta() int { return 2 }

// The noise kernel.
type noise struct{}

var Noise noise

func (n noise) Observe(x []float64) float64 {
	return 0.01 * kernel.UniformNoise.Observe(x)
}

func (noise) NTheta() int { return 1 }
