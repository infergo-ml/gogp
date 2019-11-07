package kernel

import "bitbucket.org/dtolpin/infergo/ad"

type ConstantNoise float64

func (nk ConstantNoise) Observe(x []float64) float64 {
	if ad.Called() {
		ad.Enter()
	} else {
		ad.Setup(x)
	}
	return ad.Return(ad.Value(float64(nk)))
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
	return ad.Return(&x[0])
}

func (uniformNoise) NTheta() int {
	return 1
}
