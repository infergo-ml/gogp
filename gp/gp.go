package gp

import (
	"bitbucket.org/dtolpin/gogp/kernel/ad"
	"bitbucket.org/dtolpin/infergo/model"
	"fmt"
	"gonum.org/v1/gonum/mat"
	"math"
)

// Type GP is the barebone implementation of GP.
type GP struct {
	NTheta, NNoiseTheta, NDim int         // dimensions
	Kernel, NoiseKernel       model.Model // kernel
	Theta, NoiseTheta         []float64   // kernel parameters

	// inputs
	X [][]float64 // locations, for computing covariances
	Y []float64   // observations

	// Cached computations
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

	if gp.Theta == nil {
		gp.Theta = make([]float64, gp.NTheta)
	}

	if gp.NoiseTheta == nil {
		gp.NoiseTheta = make([]float64, gp.NNoiseTheta)
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

// Absorb absorbs observations into the process.
func (gp *GP) Absorb(x [][]float64, y []float64) (err error) {
	// Set the defaults
	gp.defaults()

	// Remember the inputs
	gp.X, gp.Y = x, y
	// K's gradient by parameters and inputs
	gp.dK = make([]*mat.SymDense,
		gp.NTheta+gp.NNoiseTheta+gp.NDim*len(x))

	if len(x) == 0 {
		// No observations
		return nil
	}

	// Covariance matrix
	K := mat.NewSymDense(len(x), nil)

	for i := range gp.dK {
		gp.dK[i] = mat.NewSymDense(len(x), nil)
		gp.dK[i].Zero()
	}
	kargs := make([]float64, gp.NTheta+2*gp.NDim)
	nkargs := make([]float64, gp.NNoiseTheta+gp.NDim)
	copy(kargs, gp.Theta)
	copy(nkargs, gp.NoiseTheta)
	for i := 0; i != len(x); i++ {
		copy(kargs[gp.NTheta:], x[i])
		for j := i; j != len(x); j++ {
			copy(kargs[gp.NTheta+gp.NDim:], x[j])
			k := gp.Kernel.Observe(kargs)
			kgrad := model.Gradient(gp.Kernel)
			gp.addTodK(i, j, 0, 0, gp.NTheta, kgrad)
			gp.addTodK(i, j,
				gp.NTheta+gp.NNoiseTheta+i*gp.NDim,
				gp.NTheta,
				gp.NDim,
				kgrad)
			gp.addTodK(i, j,
				gp.NTheta+gp.NNoiseTheta+j*gp.NDim,
				gp.NTheta+gp.NDim,
				gp.NDim,
				kgrad)
			if j == i {
				// Diagonal, add noise
				copy(nkargs[gp.NNoiseTheta:], x[j])
				n := gp.NoiseKernel.Observe(nkargs)
				ngrad := model.Gradient(gp.NoiseKernel)
				gp.addTodK(i, j, gp.NTheta, 0, gp.NNoiseTheta, ngrad)
				gp.addTodK(i, j,
					gp.NTheta+gp.NNoiseTheta+j*gp.NDim,
					gp.NNoiseTheta,
					gp.NDim,
					ngrad)
				k += n
			}
			K.SetSym(i, j, k)
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

// LML computes log marginal likelihood of the kernel given the
// absorbed observations (GPML:5.8):
//   L = −½ log|Σ| − ½ y^⊤ α − n/2 log(2π), where α = Σ^-1 y
func (gp *GP) LML() float64 {
	lml := 0.
	if len(gp.X) == 0 {
		return lml
	}
	lml -= 0.5 * float64(len(gp.X)) * math.Log(2*math.Pi)
	lml -= 0.5 * gp.l.LogDet()
	lml -= 0.5 * mat.Dot(mat.NewVecDense(len(gp.Y), gp.Y), gp.alpha)
	return lml
}

// Produce computes predictions.
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
	kargs := make([]float64, gp.NTheta+2*gp.NDim)
	copy(kargs, gp.Theta)
	for i := 0; i != len(x); i++ {
		copy(kargs[gp.NTheta:], x[i])
		copy(kargs[gp.NTheta+gp.NDim:], x[i])
		k := gp.Kernel.Observe(kargs)
		model.DropGradient(gp.Kernel)
		variance.SetVec(i, k)
	}

	// Mean and covariance are computed from observations
	// if available
	if len(gp.X) > 0 {
		Kstar := mat.NewDense(len(gp.X), len(x), nil)
		kargs := make([]float64, gp.NTheta+2*gp.NDim)
		copy(kargs, gp.Theta)
		for i := 0; i != len(gp.X); i++ {
			copy(kargs[gp.NTheta:], gp.X[i])
			for j := 0; j != len(x); j++ {
				copy(kargs[gp.NTheta+gp.NDim:], x[j])
				k := gp.Kernel.Observe(kargs)
				model.DropGradient(gp.Kernel)
				Kstar.Set(i, j, k)
			}
		}

		mean.MulVec(Kstar.T(), gp.alpha)

		v := mat.NewDense(len(gp.X), len(x), nil)
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
// The model can be used on its own or as a part of a larger
// model, to infer the parameters.

// Observe computes log marginal likelihood of the parameters
// given the data. The input is contatenation of log-transformed
// hyperparameters, input locations, and input values.
//
// Optionally, the input can be only log-transformed
// hyperparameters, and then
// * only hyperparameters are inferred;
// * inputs must be assigned to fields X, Y of gp.
func (gp *GP) Observe(x []float64) float64 {
	// Transform
	for i := 0; i != gp.NTheta+gp.NNoiseTheta; i++ {
		x[i] = math.Exp(x[i])
	}

	// Destructure
	x_ := x
	gp.Theta = model.Shift(&x_, gp.NTheta)
	gp.NoiseTheta = model.Shift(&x_, gp.NNoiseTheta)
	withInputs := len(x_) > 0
	if withInputs {
		// Inputs are inferred as well as parameters,
		// normally as a part of a larger model with priors
		// on inputs.
		n := len(x_) / (gp.NDim + 1)
		gp.X = make([][]float64, n)
		for i := range gp.X {
			gp.X[i] = model.Shift(&x_, gp.NDim)
		}
		gp.Y = model.Shift(&x_, n)
	}
	if len(x_) != 0 {
		panic("len(x)")
	}

	gp.Absorb(gp.X, gp.Y)
	if !withInputs {
		gp.dK = gp.dK[:gp.NTheta+gp.NNoiseTheta]
	}

	// Restore
	for i := 0; i != gp.NTheta+gp.NNoiseTheta; i++ {
		x[i] = math.Log(x[i])
	}

	return gp.LML()
}

// Gradient computes the gradient of the log-likelihood with
// respect to the parameters and the input data locations
// (GPML:5.9):
//   ∇L = ½ tr((α α^⊤ - Σ^−1) ∂Σ/∂θ), where α = Σ^-1 y
func (gp *GP) Gradient() []float64 {
	var (
		grad       []float64
		withInputs bool
	)
	switch {
	case len(gp.dK) == gp.NTheta+gp.NNoiseTheta:
		// optimizimg hyperparameters only
		grad = make([]float64, gp.NTheta+gp.NNoiseTheta)
	case len(gp.dK) == gp.NTheta+
		gp.NNoiseTheta+
		len(gp.X)*gp.NDim:
		// optimizing everything
		grad = make([]float64,
			gp.NTheta+gp.NNoiseTheta+
				len(gp.X)*(gp.NDim+1))
		withInputs = true
	default:
		panic("len(gp.dK)")
	}

	if len(gp.X) == 0 {
		return grad
	}

	// Gradient by parameters (and possibly input locations)
	for i := range gp.dK {
		// α α^⊤ ∂Σ/∂θ
		a := mat.NewDense(len(gp.Y), len(gp.Y), nil)
		a.Mul(gp.alpha, gp.alpha.T())
		b := mat.NewDense(len(gp.Y), len(gp.Y), nil)
		b.Mul(a, gp.dK[i])

		// Σ^−1 ∂Σ/∂θ
		gp.l.SolveTo(a, gp.dK[i])

		// (α α^⊤ - Σ^−1) ∂Σ/∂θ
		c := mat.NewDense(len(gp.Y), len(gp.Y), nil)
		c.Sub(b, a)

		grad[i] = 0.5 * mat.Trace(c)
	}

	if withInputs {
		// Gradient by inputs
		for i := range gp.Y {
			grad[len(gp.dK)+i] = -gp.alpha.AtVec(i)
		}
	}
	return grad
}
