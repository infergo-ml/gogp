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
	var c1 float64
	ad.Assignment(&c1, &x[0])
	var c2 float64
	ad.Assignment(&c2, &x[1])
	var l1 float64
	ad.Assignment(&l1, &x[2])
	var l2 float64
	ad.Assignment(&l2, &x[3])
	var p float64
	ad.Assignment(&p, &x[4])
	var xa float64
	ad.Assignment(&xa, &x[5])
	var xb float64
	ad.Assignment(&xb, &x[6])

	return ad.Return(ad.Arithmetic(ad.OpAdd, ad.Arithmetic(ad.OpMul, &c1, ad.Call(func(_ []float64) {
		kernel.Matern52.Cov(0, 0, 0)
	}, 3, &l1, &xa, &xb)), ad.Arithmetic(ad.OpMul, &c2, ad.Call(func(_ []float64) {
		kernel.Periodic.Cov(0, 0, 0, 0)
	}, 4, &l2, &p, &xa, &xb))))
}

func (simil) NTheta() int	{ return 5 }

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
