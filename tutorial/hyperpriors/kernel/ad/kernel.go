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
	const (
		c1	= iota
		c2
		l1
		l2
		p
		xa
		xb
	)

	return ad.Return(ad.Arithmetic(ad.OpAdd, ad.Arithmetic(ad.OpMul, &x[c1], ad.Call(func(_ []float64) {
		kernel.Matern52.Cov(0, 0, 0)
	}, 3, &x[l1], &x[xa], &x[xb])), ad.Arithmetic(ad.OpMul, &x[c2], ad.Call(func(_ []float64) {
		kernel.Periodic.Cov(0, 0, 0, 0)
	}, 4, &x[l2], &x[p], &x[xa], &x[xb]))))
}

func (simil) NTheta() int	{ return 5 }

type noise struct{}

var Noise noise

func (n noise) Observe(x []float64) float64 {
	if ad.Called() {
		ad.Enter()
	} else {
		ad.Setup(x)
	}
	return ad.Return(ad.Call(func(_ []float64) {
		kernel.UniformNoise.Observe(x)
	}, 0))
}

func (noise) NTheta() int	{ return 1 }
