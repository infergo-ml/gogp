package gp

import (
	"bitbucket.org/dtolpin/gogp/kernel/ad"
	"bitbucket.org/dtolpin/infergo/ad"
	. "bitbucket.org/dtolpin/infergo/model"
	"fmt"
	"gonum.org/v1/gonum/mat"
	"math"
)

type GP struct {
	NTheta, NNoiseTheta, NDim int       // dimensions
	Kernel, NoiseKernel       Model     // kernel
	Theta, NoiseTheta         []float64 // kernel parameters
	Parallel                  bool      // parallelize covariances

	// Cached computations
	x     [][]float64     // inputs, for computing covariances
	l     mat.Cholesky    // Cholesky decomposition of K
	alpha *mat.VecDense   // K^-1 y
	dK    []*mat.SymDense // gradient of K
}

// Default noise, present for numerical stability; can
// be zeroed by using ConstantNoise(0.) as the noise
// kernel.
const nonoise = 1E-10

func (gp *GP) defaults() {
	if gp.NoiseKernel == nil {
		gp.NoiseKernel = kernel.ConstantNoise(nonoise)
	}

	if gp.Parallel {
		// If multithreading-safe mode is not on, it makes
		// little sense to parallelize running the kernels.
		if !ad.IsMTSafe() {
			gp.Parallel = false
		}

		// We cannot risk running kernels in parallel with elemental
		// models; this should be a rare case anyway.
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

// addTodK adds gradient components to the corresponding elements of
// dK.
func (gp *GP) addTodK(
	i, j int,
	ipar0, jpar0 int,
	narg int,
	grad []float64) {
	jpar := jpar0
	for ipar := ipar0; ipar != ipar0+narg; ipar++ {
		gp.dK[ipar].SetSym(i, j, gp.dK[ipar].At(i, j)+
			grad[jpar])
		jpar++
	}
}

// Absorb absorbs observations into the process
func (gp *GP) Absorb(x [][]float64, y []float64) (err error) {
	// Set the defaults
	gp.defaults()

	if len(x) == 0 {
		// No observations
		return nil
	}

	// Remember the input coordinates for computing covariances
	// with predictions
	gp.x = x

	// Covariance matrix
	K := mat.NewSymDense(len(x), nil)

	// K's gradient by parameters and inputs
	gp.dK = make([]*mat.SymDense,
		gp.NTheta+gp.NNoiseTheta+gp.NDim*len(x))
	for i := range gp.dK {
		gp.dK[i] = mat.NewSymDense(len(x), nil)
		gp.dK[i].Zero()
	}

	if gp.Parallel {
		// TODO
		panic("Parallel not implemented yet.")
	} else {
		kargs := make([]float64, gp.NTheta+2*gp.NDim)
		nkargs := make([]float64, gp.NNoiseTheta+gp.NDim)
		copy(kargs, gp.Theta)
		copy(nkargs, gp.NoiseTheta)
		for i := 0; i != len(x); i++ {
			// Diagonal, includes the noise
			copy(kargs[gp.NTheta:], x[i])
			copy(kargs[gp.NTheta+gp.NDim:], x[i])
			copy(nkargs[gp.NNoiseTheta:], x[i])

			k := gp.Kernel.Observe(kargs)
			kgrad := Gradient(gp.Kernel)
			gp.addTodK(i, i, 0, 0, gp.NTheta, kgrad)
			gp.addTodK(i, i,
				gp.NTheta+gp.NNoiseTheta+i*gp.NDim,
				gp.NTheta,
				gp.NDim,
				kgrad)
			// We add to the same components twice,
			// and for stationary kernels the derivatives
			// cancel each other.
			gp.addTodK(i, i,
				gp.NTheta+gp.NNoiseTheta+i*gp.NDim,
				gp.NTheta+gp.NDim,
				gp.NDim,
				kgrad)

			n := gp.NoiseKernel.Observe(nkargs)
			ngrad := Gradient(gp.NoiseKernel)
			gp.addTodK(i, i, gp.NTheta, 0, gp.NNoiseTheta, ngrad)
			gp.addTodK(i, i,
				gp.NTheta+gp.NNoiseTheta+i*gp.NDim,
				gp.NNoiseTheta,
				gp.NDim,
				ngrad)

			K.SetSym(i, i, k+n)

			// Off-diagonal, symmetric
			for j := i + 1; j != len(x); j++ {
				copy(kargs[gp.NTheta+gp.NDim:], x[j])
				k := gp.Kernel.Observe(kargs)
				kgrad := Gradient(gp.Kernel)
				gp.addTodK(i, j, 0, 0, gp.NTheta, kgrad)
				gp.addTodK(i, j,
					gp.NTheta+gp.NNoiseTheta+i*gp.NDim,
					gp.NTheta,
					gp.NDim,
					kgrad)
				gp.addTodK(i, i,
					gp.NTheta+gp.NNoiseTheta+j*gp.NDim,
					gp.NTheta+gp.NDim,
					gp.NDim,
					kgrad)

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
		panic("Parallel not implemented yet.")
	} else {
		kargs := make([]float64, gp.NTheta+2*gp.NDim)
		copy(kargs, gp.Theta)
		for i := 0; i != len(x); i++ {
			copy(kargs[gp.NTheta:], x[i])
			copy(kargs[gp.NTheta+gp.NDim:], x[i])
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
			panic("Parallel not implemented yet.")
		} else {
			kargs := make([]float64, gp.NTheta+2*gp.NDim)
			copy(kargs, gp.Theta)
			for i := 0; i != len(gp.x); i++ {
				copy(kargs[gp.NTheta:], gp.x[i])
				for j := 0; j != len(x); j++ {
					copy(kargs[gp.NTheta+gp.NDim:], x[j])
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
//   L = −½ log|Σ| − ½ y^⊤ α − n/2 log(2π), where α = Σ^-1 y
func (gp *GP) Observe(x_ []float64) float64 {
	// Destructure
	gp.Theta = Shift(&x_, gp.NTheta)
	gp.NoiseTheta = Shift(&x_, gp.NNoiseTheta)
	n := len(x_) / (gp.NDim + 1)
	x := make([][]float64, n)
	for i := range x {
		x[i] = Shift(&x_, gp.NDim)
	}
	y := x_[len(x_) - n:]

	gp.Absorb(x, y)

	// Compute L
	ll := -float64(n)*math.Log(2*math.Pi)
	ll -= 0.5*gp.l.LogDet()
	ll -= 0.5*mat.Dot(mat.NewVecDense(n, y), gp.alpha)
	return ll
}

// gradll computes the gradient of the log-likelihood with respect
// to the parameters and the input data locations (GPML:5.9):
//   ∇L = ½ tr((α α^⊤ - Σ^−1) ∂Σ/∂θ), where α = Σ^-1 y
func (gp *GP) Gradient() []float64 {
	grad := make([]float64, len(gp.dK))
	for i := range gp.dK {
		// α α^⊤ ∂Σ/∂θ
		a := mat.NewDense(len(gp.x), len(gp.x), nil)
		a.Mul(gp.alpha, gp.alpha.T())
		b := mat.NewDense(len(gp.x), len(gp.x), nil)
		b.Mul(a, gp.dK[i])

		// Σ^−1 ∂Σ/∂θ
		gp.l.SolveTo(a, gp.dK[i])

		// (α α^⊤ - Σ^−1) ∂Σ/∂θ
		c := mat.NewDense(len(gp.x), len(gp.x), nil)
		c.Sub(b, a)

		grad[i] = 0.5 * mat.Trace(c)
	}
	return grad
}
