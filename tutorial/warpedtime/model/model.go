package model

import (
	. "bitbucket.org/dtolpin/infergo/dist"
)

type Priors struct{
	step []float64 // steps between consequent points
}

func (m *Priors) Observe(x []float64) float64 {
	const (
		c = iota // output scale
		l        // length scale
		s        // noise
		i0       // first input location
	)

	n := len(x[i0:])/2
	if len(m.step) != n - 1 {
		if n > 1 {
			// First call, memoize initial distances between inputs
			m.step = make([]float64, n - 1)
			for i := range m.step {
				m.step[i], _ = x[i0+i+1] - x[i0+i], true
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
	ll += Normal.Logp(0, 1, x[s])

	//  We allow input locations to move slightly.
	for i := range m.step {
		ll += Normal.Logp(1, 0.2, (x[i0 + i + 1] - x[i0 + i])/m.step[i])
	}

	return ll
}
