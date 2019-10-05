package gp

import (
	"bitbucket.org/dtolpin/gogp/kernel/ad"
	"bitbucket.org/dtolpin/infergo/model"
	"fmt"
	"gonum.org/v1/gonum/mat"
	"math"
)

// Type Kernel is the kernel interface, implemented by
// both covariance and noise kernels.
type Kernel interface {
	model.Model
	NTheta() int
}

// Type GP is the barebone implementation of GP.
type GP struct {
	NDim                   int       // dimensions
	Simil, Noise           Kernel    // kernel
	ThetaSimil, ThetaNoise []float64 // kernel parameters

	// inputs
	X [][]float64 // inputs, for computing covariances
	Y []float64   // outputs

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
	if gp.Noise == nil {
		gp.Noise = kernel.ConstantNoise(nonoise)
	}

	if len(gp.ThetaSimil) == 0 {
		gp.ThetaSimil = make([]float64, gp.Simil.NTheta())
	}

	if len(gp.ThetaNoise) == 0 {
		gp.ThetaNoise = make([]float64, gp.Noise.NTheta())
	}
}

// addTodK adds gradient components fromto the corresponding
// elements of dK.
func (gp *GP) addTodK(
	i, j int,
	ipar0 /* over dK */, jpar0 /* over grad */ int,
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
		gp.Simil.NTheta()+gp.Noise.NTheta()+gp.NDim*len(x))

	if len(x) == 0 {
		// No observations
		return nil
	}

	// Similariance matrix
	K := mat.NewSymDense(len(x), nil)

	for i := range gp.dK {
		gp.dK[i] = mat.NewSymDense(len(x), nil)
		gp.dK[i].Zero()
	}
	kargs := make([]float64, gp.Simil.NTheta()+2*gp.NDim)
	nkargs := make([]float64, gp.Noise.NTheta()+gp.NDim)
	copy(kargs, gp.ThetaSimil)
	copy(nkargs, gp.ThetaNoise)
	for i := 0; i != len(x); i++ {
		copy(kargs[gp.Simil.NTheta():], x[i])
		for j := i; j != len(x); j++ {
			copy(kargs[gp.Simil.NTheta()+gp.NDim:], x[j])
			k := gp.Simil.Observe(kargs)
			kgrad := model.Gradient(gp.Simil)
			gp.addTodK(i, j, 0, 0, gp.Simil.NTheta(), kgrad)
			gp.addTodK(i, j,
				gp.Simil.NTheta()+gp.Noise.NTheta()+i*gp.NDim,
				gp.Simil.NTheta(),
				gp.NDim,
				kgrad)
			gp.addTodK(i, j,
				gp.Simil.NTheta()+gp.Noise.NTheta()+j*gp.NDim,
				gp.Simil.NTheta()+gp.NDim,
				gp.NDim,
				kgrad)
			if j == i { // Diagonal, add noise
				copy(nkargs[gp.Noise.NTheta():], x[j])
				n := gp.Noise.Observe(nkargs)
				ngrad := model.Gradient(gp.Noise)
				gp.addTodK(i, j, gp.Simil.NTheta(), 0, gp.Noise.NTheta(), ngrad)
				gp.addTodK(i, j,
					gp.Simil.NTheta()+gp.Noise.NTheta()+j*gp.NDim,
					gp.Noise.NTheta(),
					gp.NDim,
					ngrad)
				k += n
			}
			K.SetSym(i, j, k)
		}
	}

	if !gp.l.Factorize(K) {
		return fmt.Errorf("Factorize(%v)", mat.Formatted(K))
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
	kargs := make([]float64, gp.Simil.NTheta()+2*gp.NDim)
	copy(kargs, gp.ThetaSimil)
	for i := 0; i != len(x); i++ {
		copy(kargs[gp.Simil.NTheta():], x[i])
		copy(kargs[gp.Simil.NTheta()+gp.NDim:], x[i])
		k := gp.Simil.Observe(kargs)
		model.DropGradient(gp.Simil)
		variance.SetVec(i, k)
	}

	// Mean and covariance are computed from observations
	// if available
	if len(gp.X) > 0 {
		Kstar := mat.NewDense(len(gp.X), len(x), nil)
		kargs := make([]float64, gp.Simil.NTheta()+2*gp.NDim)
		copy(kargs, gp.ThetaSimil)
		for i := 0; i != len(gp.X); i++ {
			copy(kargs[gp.Simil.NTheta():], gp.X[i])
			for j := 0; j != len(x); j++ {
				copy(kargs[gp.Simil.NTheta()+gp.NDim:], x[j])
				k := gp.Simil.Observe(kargs)
				model.DropGradient(gp.Simil)
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
// The model can be used on its own or, as a part of a larger
// model, to infer the parameters.

// Observe computes log marginal likelihood of the parameters
// given the data. The argument is contatenation of log-transformed
// hyperparameters, inputs, and outputs.
//
// Optionally, the input can be only log-transformed
// hyperparameters, and then
// * only hyperparameters are inferred;
// * inputs must be assigned to fields X, Y of gp.
func (gp *GP) Observe(x []float64) float64 {
	gp.defaults()
	// Transform
	for i := 0; i != gp.Simil.NTheta()+gp.Noise.NTheta(); i++ {
		x[i] = math.Exp(x[i])
	}

	// Destructure
	x_ := x
	copy(gp.ThetaSimil, model.Shift(&x_, gp.Simil.NTheta()))
	copy(gp.ThetaNoise, model.Shift(&x_, gp.Noise.NTheta()))
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

	err := gp.Absorb(gp.X, gp.Y)
	if err != nil {
		panic(err)
	}

	if !withInputs {
		// If inputs are not inferred, we drop dK components
		// corresponding to derivatives by inputs and outputs.
		gp.dK = gp.dK[:gp.Simil.NTheta()+gp.Noise.NTheta()]
	}

	// Restore
	for i := 0; i != gp.Simil.NTheta()+gp.Noise.NTheta(); i++ {
		x[i] = math.Log(x[i])
	}

	return gp.LML()
}

// Gradient computes the gradient of the log-likelihood with
// respect to the parameters and the inputs (GPML:5.9):
//   ∇L = ½ tr((α α^⊤ - Σ^−1) ∂Σ/∂θ), where α = Σ^-1 y
func (gp *GP) Gradient() []float64 {
	var (
		grad       []float64
		withInputs bool
	)
	switch {
	case len(gp.dK) == gp.Simil.NTheta()+gp.Noise.NTheta():
		// optimizimg hyperparameters only
		grad = make([]float64, gp.Simil.NTheta()+gp.Noise.NTheta())
	case len(gp.dK) == gp.Simil.NTheta()+gp.Noise.NTheta()+
		len(gp.X)*gp.NDim:
		// optimizing everything
		grad = make([]float64,
			gp.Simil.NTheta()+gp.Noise.NTheta()+len(gp.X)*(gp.NDim+1))
		withInputs = true
	default:
		// cannot happen
		panic("len(gp.dK)")
	}

	if len(gp.X) == 0 {
		// no inputs, return zero gradient
		return grad
	}

	// Gradient by parameters (and possibly inputs)
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
		// Gradient by outputs
		for i := range gp.Y {
			grad[len(gp.dK)+i] = -gp.alpha.AtVec(i)
		}
	}
	return grad
}
