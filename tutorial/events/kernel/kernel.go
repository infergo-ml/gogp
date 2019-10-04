package kernel

import (
	"bitbucket.org/dtolpin/gogp/kernel"
)

// The similarity kernel. When two points are on different sides
// of an event boundary, the similarity between the points is
// scaled down by the event's discount factor.
type Simil struct {
	Events [][]float64
}

func (s *Simil) Observe(x []float64) float64 {
	// Parameters
	const (
		c = iota
		l
		a
		b
	)

	// Event fields
	const (
		from = iota
		to
		discount
	)

	k := x[c] * kernel.Matern52.Observe(x[l:])

	// Discount similarities crossing event boundaries
	for i := range s.Events {
		e := s.Events[i]
		xa, xb := x[a], x[b]
		if xa > xb {
			xa, xb = xb, xa
		}
		if xa <= e[from] && e[from] <= xb ||
			xa <= e[to] && e[to] <= xb {
			k *= e[discount]
			break
		}
	}
	return k
}

func (*Simil) NTheta() int { return 2 }

// The noise kernel, uniform noise scaled by a likely value.
// Scaling is tantamount to specifying an initial
// point in the proximity of a reasonable noise variance but
// makes the code easier to write and read.
type Noise float64

func (n Noise) Observe(x []float64) float64 {
	return float64(n) * kernel.UniformNoise.Observe(x)
}

func (Noise) NTheta() int { return 1 }
