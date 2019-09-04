package gp

import (
	. "bitbucket/org/dtolpin/infergo/model"
	"gonum.org/v1/gonum/mat"
)

type GP struct {
	NParam, NNoiseParam, NDim int       // dimensions
	Kernel, NoiseKernel       Model     // kernel
	Theta, ThetaNoise         []float64 // kernel parameters
}

// Absorb absorbs observations into the process
func (gp GP) Absorb(x [][]float64, y []float64) {
	// Covariance matrix
	K := mat.NewSymDense(len(x), nil)
	kargs := make([]float64, NParam+2*NDim)
	nkargs := make([]float64, NNoiseParam+NDim)
	copy(kargs, gp.Theta)
	copy(nkargs, gp.ThetaNoise)
	for i := 0; i != len(x); i++ {
		copy(kargs[gp.NParam:], x[i])
		copy(nkargs[gp.NNoiseParam:], x[i])

		// Diagonal, includes the noise
		copy(kargs[gp.NParam+gp.NDim:], x[i])
		k := gp.Kernel.Observe(kargs)
		n := gp.NoiseKernel.Observe(nkargs)
		// TODO: gradient
		K.Set(i, i, k+n)

		// Off-diagonal, symmetric
		for j := i + 1; i != len(x); j++ {
			copy(kargs[gp.NParam+gp.NDim:], x[j])
			k := gp.Kernel.Observe(kargs)
			// TODO: gradient
			K.SetSym(i, j, k)
		}
	}

	var L mat.Cholesky
	// TODO
}

// Produce computes predictions
func (gp GP) Produce(x [][]float64) (mu, sigma []float64) {
}

// Observe and Gradient implement Infergo's ElementalModel.
// The model can be used, on its own or as a part of a larger
// model, to infer the parameters.

// Observe computes log-likelihood of the parameters given the data
// (GPML:5.8):
//   L = −½ log|Σ| − ½ y^⊤ Σ^−1 y − n/2 log(2π)
func (gp GP) Observe(x []float64) float64 {
	return 0.
}

// gradll computes the gradient of the log-likelihood with respect
// to the parameters and the input data locations (GPML:5.9):
//   ∇L = ½ tr((α α^⊤ - Σ^−1) ∂Σ/∂θ), where α = Σ^-1 y
func (gp GP) Gradient() []float64 {
	return []float64{}
}
