package kernel

import "bitbucket.org/dtolpin/infergo/ad"

type ConstantNoise float64

func (nk ConstantNoise) Observe(x []float64) float64 {
	if ad.Called() {
		ad.Enter()
	} else {
		ad.Setup(x)
	}
	return ad.Return(ad.Call(func(_ []float64) {
		nk.Var()
	}, 0))
}

func (nk ConstantNoise) Var() float64 {
	if ad.Called() {
		ad.Enter()
	} else {
		panic("Var called outside Observe")
	}
	var std float64
	ad.Assignment(&std, ad.Value(float64(nk)))
	return ad.Return(ad.Arithmetic(ad.OpMul, &std, &std))
}

func (ConstantNoise) NTheta() int {
	return 0
}

type uniformNoise struct{}

var UniformNoise uniformNoise

func (nk uniformNoise) Observe(x []float64) float64 {
	if ad.Called() {
		ad.Enter()
	} else {
		ad.Setup(x)
	}
	return ad.Return(ad.Call(func(_ []float64) {
		nk.Var(0)
	}, 1, &x[0]))
}

func (nk uniformNoise) Var(std float64) float64 {
	if ad.Called() {
		ad.Enter(&std)
	} else {
		panic("Var called outside Observe")
	}
	return ad.Return(ad.Arithmetic(ad.OpMul, &std, &std))
}

func (uniformNoise) NTheta() int {
	return 1
}
