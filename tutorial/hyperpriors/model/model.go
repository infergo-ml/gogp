package model

import (
	. "bitbucket.org/dtolpin/infergo/dist"
	"math"
)

type Priors struct{}

func (*Priors) Observe(x []float64) float64 {
	const (
		c1 = iota // trend scale
		c2        // season scale
		l1        // trend length scale
		l2        // season length scale
		p         // season period
		s         // noise
	)

	ll := 0.

	// trend weight is a number between 0 and 1
	ll += Normal.Logp(-1, 2, x[c1])

	// seasonality weight is normally lower than trend weight
	ll += Normal.Logp(x[c1]-math.Log(2), 1, x[c2])

	// length scale is around 1, in wide margins
	ll += Normal.Logp(0, 2, x[l1])
	ll += Normal.Logp(0, 2, x[l2])

	// the period is scaled by 10 in the kernel, the actual
	// period is 8. We pretend we know the period approximately.
	ll += Normal.Logp(0, 1, x[p])

	// The noise is scaled by 0.01 in the kernel.
	ll += Normal.Logp(0, 1, x[s])

	return ll
}
