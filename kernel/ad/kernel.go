package kernel

import (
	"bitbucket.org/dtolpin/infergo/ad"
	"math"
)

type normal struct{}

var Normal normal

func (k normal) Observe(x []float64) float64 {
	if ad.Called() {
		ad.Enter()
	} else {
		ad.Setup(x)
	}
	return ad.Return(ad.Call(func(_ []float64) {
		k.Cov(0, 0, 0)
	}, 3, &x[0], &x[1], &x[2]))
}

func (normal) NTheta() int {
	return 1
}

func (normal) Cov(l, xa, xb float64) float64 {
	if ad.Called() {
		ad.Enter(&l, &xa, &xb)
	} else {
		panic("Cov called outside Observe.")
	}
	var d float64
	ad.Assignment(&d, ad.Arithmetic(ad.OpDiv, (ad.Arithmetic(ad.OpSub, &xa, &xb)), &l))
	return ad.Return(ad.Elemental(math.Exp, ad.Arithmetic(ad.OpDiv, ad.Arithmetic(ad.OpMul, ad.Arithmetic(ad.OpNeg, &d), &d), ad.Value(2))))
}

type periodic struct{}

var Periodic periodic

func (k periodic) Observe(x []float64) float64 {
	if ad.Called() {
		ad.Enter()
	} else {
		ad.Setup(x)
	}
	return ad.Return(ad.Call(func(_ []float64) {
		k.Cov(0, 0, 0, 0)
	}, 4, &x[0], &x[1], &x[2], &x[3]))
}

func (periodic) NTheta() int {
	return 2
}

func (periodic) Cov(l, p, xa, xb float64) float64 {
	if ad.Called() {
		ad.Enter(&l, &p, &xa, &xb)
	} else {
		panic("Cov called outside Observe.")
	}
	var d float64
	ad.Assignment(&d, ad.Arithmetic(ad.OpDiv, ad.Elemental(math.Sin, ad.Arithmetic(ad.OpDiv, ad.Arithmetic(ad.OpMul, ad.Value(math.Pi), ad.Elemental(math.Abs, ad.Arithmetic(ad.OpSub, &xa, &xb))), &p)), &l))
	return ad.Return(ad.Elemental(math.Exp, ad.Arithmetic(ad.OpMul, ad.Arithmetic(ad.OpMul, ad.Value(-2), &d), &d)))
}

const (
	sqrt3 = 1.7320508075688772
	sqrt5 = 2.2360679774997900
)

type matern32 struct{}

var Matern32 matern32

func (k matern32) Observe(x []float64) float64 {
	if ad.Called() {
		ad.Enter()
	} else {
		ad.Setup(x)
	}
	return ad.Return(ad.Call(func(_ []float64) {
		k.Cov(0, 0, 0)
	}, 3, &x[0], &x[1], &x[2]))
}

func (matern32) NTheta() int {
	return 1
}

func (matern32) Cov(l, xa, xb float64) float64 {
	if ad.Called() {
		ad.Enter(&l, &xa, &xb)
	} else {
		panic("Cov called outside Observe.")
	}
	var d float64
	ad.Assignment(&d, ad.Arithmetic(ad.OpDiv, ad.Elemental(math.Abs, ad.Arithmetic(ad.OpSub, &xa, &xb)), &l))
	return ad.Return(ad.Arithmetic(ad.OpMul, (ad.Arithmetic(ad.OpAdd, ad.Value(1), ad.Arithmetic(ad.OpMul, ad.Value(sqrt3), &d))), ad.Elemental(math.Exp, ad.Arithmetic(ad.OpMul, ad.Value(-1.7320508075688772), &d))))
}

type matern52 struct{}

var Matern52 matern52

func (k matern52) Observe(x []float64) float64 {
	if ad.Called() {
		ad.Enter()
	} else {
		ad.Setup(x)
	}
	return ad.Return(ad.Call(func(_ []float64) {
		k.Cov(0, 0, 0)
	}, 3, &x[0], &x[1], &x[2]))
}

func (matern52) NTheta() int {
	return 1
}

func (matern52) Cov(l, xa, xb float64) float64 {
	if ad.Called() {
		ad.Enter(&l, &xa, &xb)
	} else {
		panic("Cov called outside Observe.")
	}
	var d float64
	ad.Assignment(&d, ad.Arithmetic(ad.OpDiv, ad.Elemental(math.Abs, ad.Arithmetic(ad.OpSub, &xa, &xb)), &l))
	return ad.Return(ad.Arithmetic(ad.OpMul, (ad.Arithmetic(ad.OpAdd, ad.Arithmetic(ad.OpAdd, ad.Value(1), ad.Arithmetic(ad.OpMul, ad.Value(sqrt5), &d)), ad.Arithmetic(ad.OpMul, ad.Arithmetic(ad.OpMul, ad.Value(1), &d), &d))), ad.Elemental(math.Exp, ad.Arithmetic(ad.OpMul, ad.Value(-2.23606797749979), &d))))
}
