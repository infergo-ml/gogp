package kernel

import (
	"bitbucket.org/dtolpin/gogp/kernel"
)

type simil struct{}

var Simil simil

func (simil) Observe(x []float64) float64 {
	return x[0] * kernel.Normal.Observe(x[1:])
}

func (simil) NTheta() int {
	return 2
}
