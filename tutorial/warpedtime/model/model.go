package model

import (
	. "bitbucket.org/dtolpin/infergo/dist"
	"math"
)

type Priors struct {
	LogSigma float64   // log standard deviation of relative step
	step     []float64 // steps between consequent points
}

func (m *Priors) Observe(x []float64) float64 {
	const (
		c  = iota // output scale
		l         // length scale
		s         // noise
		i0        // first input
	)

	n := len(x[i0:]) / 2
	if len(m.step) != n-1 {
		if n > 1 {
			// First call, memoize initial distances between inputs
			m.step = make([]float64, n-1)
			for i := range m.step {
				// For use with the tutorial, we hide the assignment
				// from automatic differentation by making it a parallel
				// non float64 assignment (which is not differentiated).
				// Otherwise, the values of m.step elements will be
				// restored by the backward pass.
				//
				// In a practical implementation, steps should rather be
				// pre-computed before inference.
				m.step[i], _ = x[i0+i+1]-x[i0+i], true
			}
		} else {
			m.step = nil
		}
	}

	ll := 0.

	// Output scale is mostly less than 1.
	ll += Normal.Logp(-1, 1, x[c])

	// Length scale is around 1, in wide margins.
	ll += Normal.Logp(0, 2, x[l])

	// The noise is scaled by 0.01 in the kernel.
	ll += Normal.Logp(0.5, 1, x[s])

	//  We allow inputs to move slightly.
	for i := range m.step {
		ll += Normal.Logp(1, math.Exp(m.LogSigma),
			(x[i0+i+1]-x[i0+i])/m.step[i])
	}

	return ll
}
