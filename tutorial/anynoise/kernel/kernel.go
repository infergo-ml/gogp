package kernel

import (
	"bitbucket.org/dtolpin/gogp/kernel"
)

// The similarity kernel.
type simil struct{}

var Simil simil

func (simil) Observe(x []float64) float64 {
	const (
		c  = iota // output scale
		l         // length scale
		xa        // first point
		xb        // second point
	)

	return x[c] * kernel.Matern52.Cov(x[l], x[xa], x[xb])
}

func (simil) NTheta() int { return 2 }

// The noise kernel, allocates a single parameter,
// which is used to define the noise in the priors.
type noise struct{}

var Noise noise

func (n noise) Observe(x []float64) float64 {
	return 1e-5 // added for numerical stability
}

func (noise) NTheta() int { return 1 }
