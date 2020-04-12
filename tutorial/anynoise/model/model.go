package model

import (
	. "bitbucket.org/dtolpin/infergo/dist"
	"math"
)

type Priors struct {
	Y []float64 // noisy outputs
}

func (m *Priors) Observe(x []float64) float64 {
	const (
		s  = iota // noise
		c         // output scale
		l         // length scale
		i0        // first input
	)

	n := len(x[i0:]) / 2
	if len(m.Y) != n {
		// First call, memoize initial outputs
		m.Y = make([]float64, n)
		copy(m.Y, x[i0+n:])
	}

	ll := 0.

	// The noise standard deviation is less than 1, in wide
	// margins.
	ll += Normal.Logp(-1, 2, x[s])

	// Output scale is mostly less than 1.
	ll += Normal.Logp(-1, 1, x[c])

	// Length scale is around 1, in wide margins.
	ll += Normal.Logp(0, 2, x[l])

	// Instead of Gaussian, we assume Laplacian noise.
	for i := range m.Y {
		ll += Expon.Logp(1/math.Exp(x[s]),
			math.Abs(m.Y[i]-x[i0+n+i]))
	}

	return ll
}
