package gp

import (
	"bitbucket.org/dtolpin/gogp/kernel/ad"
	. "bitbucket.org/dtolpin/infergo/model"
	"fmt"
	"gonum.org/v1/gonum/mat"
	"math"
)

type GP struct {
	NParam, NNoiseParam, NDim int       // dimensions
	Kernel, NoiseKernel       Model     // kernel
	Theta, NoiseTheta         []float64 // kernel parameters
	Parallel                  bool      // parallelize covariances

	// Cached computations
	x     [][]float64
	l     mat.Cholesky
	alpha *mat.VecDense
}

// Default noise, present for numerical stability; can
// be zeroed by using ConstantNoise(0.) as the noise
// kernel.
const nonoise = 1E-10

func (gp GP) defaults() {
	if gp.NoiseKernel == nil {
		gp.NoiseKernel = kernel.ConstantNoise(nonoise)
		gp.NoiseTheta = make([]float64, 0)
	}
}

// Absorb absorbs observations into the process
func (gp GP) Absorb(x [][]float64, y []float64) (err error) {
	// Set the defaults
	gp.defaults()

	// Remember the input coordinates for computing covariances
	// with predictions
	gp.x = x

	// Covariance matrix
	K := mat.NewSymDense(len(x), nil)
	if gp.Parallel {
		// TODO
	} else {
		kargs := make([]float64, gp.NParam+2*gp.NDim)
		nkargs := make([]float64, gp.NNoiseParam+gp.NDim)
		copy(kargs, gp.Theta)
		copy(nkargs, gp.NoiseTheta)
		for i := 0; i != len(x); i++ {
			copy(kargs[gp.NParam:], x[i])
			copy(nkargs[gp.NNoiseParam:], x[i])

			// Diagonal, includes the noise
			copy(kargs[gp.NParam+gp.NDim:], x[i])
			k := gp.Kernel.Observe(kargs)
			n := gp.NoiseKernel.Observe(nkargs)
			// TODO: gradient
			K.SetSym(i, i, k+n)

			// Off-diagonal, symmetric
			for j := i + 1; i != len(x); j++ {
				copy(kargs[gp.NParam+gp.NDim:], x[j])
				k := gp.Kernel.Observe(kargs)
				// TODO: gradient
				K.SetSym(i, j, k)
			}
		}
	}

	if !gp.l.Factorize(K) {
		return fmt.Errorf("factorize")
	}

	gp.alpha = mat.NewVecDense(len(x), nil)
	err = gp.l.SolveVecTo(gp.alpha, mat.NewVecDense(len(y), y))
	if err != nil {
		return err
	}

	return
}

// Produce computes predictions
func (gp GP) Produce(x [][]float64) (
	mu, sigma []float64,
	err error,
) {
	// TODO: from prior?

	Kstar := mat.NewDense(len(gp.x), len(x), nil)
	if gp.Parallel {
		//TODO
	} else {
		kargs := make([]float64, gp.NParam+2*gp.NDim)
		copy(kargs, gp.Theta)
		for i := 0; i != len(gp.x); i++ {
			copy(kargs[gp.NParam:], gp.x[i])
			for j := 0; i != len(x); j++ {
				copy(kargs[gp.NParam+gp.NDim:], x[j])
				k := gp.Kernel.Observe(kargs)
				// TODO: drop gradient
				Kstar.Set(i, j, k)
			}
		}
	}

	var mean mat.VecDense
	mean.MulVec(Kstar, gp.alpha)
	mu = mean.RawVector().Data

	variance := mat.NewVecDense(len(x), nil)
	if gp.Parallel {
		// TODO
	} else {
		kargs := make([]float64, gp.NParam+2*gp.NDim)
		copy(kargs, gp.Theta)
		for i := 0; i != len(x); i++ {
			copy(kargs[gp.NParam:], gp.x[i])
			copy(kargs[gp.NParam+gp.NDim:], gp.x[i])
			k := gp.Kernel.Observe(kargs)
			// TODO: drop gradient
			variance.SetVec(i, k)
		}
	}

	v := mat.NewDense(len(gp.x), len(x), nil)
	if err := gp.l.SolveTo(v, Kstar); err != nil {
		return nil, nil, err
	}
	covariance := mat.NewDense(len(x), len(x), nil)
	covariance.Mul(Kstar.T(), v)

	// TODO: find an idiomatic way to write this
	sigma = variance.RawVector().Data
	for i := range sigma {
		sigma[i] = math.Sqrt(sigma[i] - covariance.At(i, i))
	}

	return mu, sigma, nil
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
