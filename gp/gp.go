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

func (gp *GP) defaults() {
	if gp.NoiseKernel == nil {
		gp.NoiseKernel = kernel.ConstantNoise(nonoise)
		gp.NoiseTheta = make([]float64, 0)
	}

	// We cannot risk running kernels in parallel with elemental
	// models; this should be a rare case anyway.
	if gp.Parallel {
		_, isElemental := gp.Kernel.(ElementalModel)
		if isElemental {
			gp.Parallel = false
		}
		_, isElemental = gp.NoiseKernel.(ElementalModel)
		if isElemental {
			gp.Parallel = false
		}
	}
}

// Absorb absorbs observations into the process
func (gp *GP) Absorb(x [][]float64, y []float64) (err error) {
	// Set the defaults
	gp.defaults()

	// Remember the input coordinates for computing covariances
	// with predictions
	gp.x = x

	// Covariance matrix
	K := mat.NewSymDense(len(x), nil)
	if gp.Parallel {
		// TODO
		return fmt.Errorf("Parallel not implemented yet.")
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
			kgrad := Gradient(gp.Kernel)
			n := gp.NoiseKernel.Observe(nkargs)
			ngrad := Gradient(gp.NoiseKernel)
			// TODO: collect gradients
			kgrad = kgrad
			ngrad = ngrad
			K.SetSym(i, i, k+n)

			// Off-diagonal, symmetric
			for j := i + 1; j != len(x); j++ {
				copy(kargs[gp.NParam+gp.NDim:], x[j])
				k := gp.Kernel.Observe(kargs)
				kgrad := Gradient(gp.Kernel)
				// TODO: collect gradients
				kgrad = kgrad
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
func (gp *GP) Produce(x [][]float64) (
	mu, sigma []float64,
	err error,
) {
	// Set the defaults
	gp.defaults()

	mean := mat.NewVecDense(len(x), nil)
	variance := mat.NewVecDense(len(x), nil)
	covariance := mat.NewDense(len(x), len(x), nil)

	// Prior variance does not depend on observations
	if gp.Parallel {
		// TODO
		return nil, nil,
			fmt.Errorf("Parallel not implemented yet.")
	} else {
		kargs := make([]float64, gp.NParam+2*gp.NDim)
		copy(kargs, gp.Theta)
		for i := 0; i != len(x); i++ {
			copy(kargs[gp.NParam:], x[i])
			copy(kargs[gp.NParam+gp.NDim:], x[i])
			k := gp.Kernel.Observe(kargs)
			DropGradient(gp.Kernel)
			variance.SetVec(i, k)
		}
	}

	// Mean and covariance are computed from observations
	// if available
	if len(gp.x) > 0 {
		Kstar := mat.NewDense(len(gp.x), len(x), nil)
		if gp.Parallel {
			// TODO
			return nil, nil,
				fmt.Errorf("Parallel not implemented yet.")
		} else {
			kargs := make([]float64, gp.NParam+2*gp.NDim)
			copy(kargs, gp.Theta)
			for i := 0; i != len(gp.x); i++ {
				copy(kargs[gp.NParam:], gp.x[i])
				for j := 0; j != len(x); j++ {
					copy(kargs[gp.NParam+gp.NDim:], x[j])
					k := gp.Kernel.Observe(kargs)
					DropGradient(gp.Kernel)
					Kstar.Set(i, j, k)
				}
			}
		}

		mean.MulVec(Kstar.T(), gp.alpha)

		v := mat.NewDense(len(gp.x), len(x), nil)
		if err := gp.l.SolveTo(v, Kstar); err != nil {
			return nil, nil, err
		}
		covariance = mat.NewDense(len(x), len(x), nil)
		covariance.Mul(Kstar.T(), v)
	} else {
		// No observations
		mean.Zero()
		covariance.Zero()
	}

	mu = make([]float64, len(x))
	for i := range mu {
		mu[i] = mean.AtVec(i)
	}

	sigma = make([]float64, len(x))
	for i := range sigma {
		sigma[i] = math.Sqrt(variance.AtVec(i) - covariance.At(i, i))
	}

	return mu, sigma, nil
}

// Observe and Gradient implement Infergo's ElementalModel.
// The model can be used, on its own or as a part of a larger
// model, to infer the parameters.

// Observe computes log-likelihood of the parameters given the data
// (GPML:5.8):
//   L = −½ log|Σ| − ½ y^⊤ Σ^−1 y − n/2 log(2π)
func (gp *GP) Observe(x []float64) float64 {
	return 0.
}

// gradll computes the gradient of the log-likelihood with respect
// to the parameters and the input data locations (GPML:5.9):
//   ∇L = ½ tr((α α^⊤ - Σ^−1) ∂Σ/∂θ), where α = Σ^-1 y
func (gp *GP) Gradient() []float64 {
	return []float64{}
}
