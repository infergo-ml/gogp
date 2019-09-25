package kernel

import (
	"bitbucket.org/dtolpin/gogp/kernel/ad"
	"bitbucket.org/dtolpin/infergo/ad"
)

type simil struct{}

var Simil simil

func (simil) Observe(x []float64) float64 {
	if ad.Called() {
		ad.Enter()
	} else {
		ad.Setup(x)
	}
	return ad.Return(ad.Arithmetic(ad.OpMul, &x[0], ad.Call(func(_ []float64) {
		kernel.Normal.Observe(x[1:])
	}, 0)))
}

func (simil) NTheta() int	{ return 2 }

type Noise float64

func (n Noise) Observe(x []float64) float64 {
	if ad.Called() {
		ad.Enter()
	} else {
		ad.Setup(x)
	}
	return ad.Return(ad.Arithmetic(ad.OpMul, ad.Value(float64(n)), ad.Call(func(_ []float64) {
		kernel.UniformNoise.Observe(x)
	}, 0)))
}

func (Noise) NTheta() int	{ return 1 }
