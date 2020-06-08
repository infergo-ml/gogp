package gp

import (
	"bitbucket.org/dtolpin/gogp/kernel/ad"
	"bitbucket.org/dtolpin/infergo/model"
	"fmt"
	"gonum.org/v1/gonum/mat"
	"math"
	"sync"
)

// Type Kernel is the kernel interface, implemented by
// both covariance and noise kernels.
type Kernel interface {
	model.Model
	NTheta() int
}

// Type GP is the barebone implementation of GP.
type GP struct {
	NDim                   int       // number of dimensions
	Simil, Noise           Kernel    // kernels
	ThetaSimil, ThetaNoise []float64 // kernel parameters

	X [][]float64 // inputs
	Y []float64   // outputs

	// When true, covariances are computed in parallel
	Parallel bool

	// Cached computations
	withObs bool            // true when observations are inferred
	l       mat.Cholesky    // Cholesky decomposition of K
	alpha   *mat.VecDense   // K^-1 y
	dK      []*mat.SymDense // gradient of K
}

// Default noise, present for numerical stability; can
// be zeroed by using ConstantNoise(0.) as the noise
// kernel.
const nonoise = 1e-5

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

// addTodK adds gradient components to the corresponding
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

const (
	withoutGradient = false
	withGradient    = true
)

// Absorb absorbs observations into the process.
func (gp *GP) Absorb(x [][]float64, y []float64) (err error) {
	// Set the defaults
	gp.defaults()
	// Remember the inputs
	gp.X, gp.Y = x, y
	// When Absorb is called directly, the gradient is not computed
	return gp.absorb(withoutGradient)
}

func (gp *GP) absorb(withGrad bool) (err error) {
	if withGrad {
		// K's gradient by parameters and inputs
		if gp.withObs {
			gp.dK = make([]*mat.SymDense,
				gp.Simil.NTheta()+gp.Noise.NTheta()+gp.NDim*len(gp.X))
		} else {
			gp.dK = make([]*mat.SymDense,
				gp.Simil.NTheta()+gp.Noise.NTheta())
		}
	}

	if len(gp.X) == 0 {
		// No observations
		return nil
	}

	// Covariance matrix
	K := mat.NewSymDense(len(gp.X), nil)

	cov := func(i, j int, kargs, nargs []float64) {
		copy(kargs[gp.Simil.NTheta()+gp.NDim:], gp.X[j])
		k := gp.Simil.Observe(kargs)
		if withGrad {
			kgrad := model.Gradient(gp.Simil)
			for i := 0; i != gp.Simil.NTheta(); i++ {
				kgrad[i] *= gp.ThetaSimil[i]
			}
			gp.addTodK(i, j, 0, 0, gp.Simil.NTheta(), kgrad)
			if gp.withObs {
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
			}
		} else {
			model.DropGradient(gp.Simil)
		}
		if j == i { // Diagonal, add noise
			copy(nargs[gp.Noise.NTheta():], gp.X[j])
			n := gp.Noise.Observe(nargs)
			if withGrad {
				ngrad := model.Gradient(gp.Noise)
				for i := 0; i != gp.Noise.NTheta(); i++ {
					ngrad[i] *= gp.ThetaNoise[i]
				}
				gp.addTodK(i, j,
					gp.Simil.NTheta(), 0, gp.Noise.NTheta(), ngrad)
				if gp.withObs {
					gp.addTodK(i, j,
						gp.Simil.NTheta()+gp.Noise.NTheta()+j*gp.NDim,
						gp.Noise.NTheta(),
						gp.NDim,
						ngrad)
				}
			} else {
				model.DropGradient(gp.Noise)
			}
			k += n
		}
		K.SetSym(i, j, k)
	}

	if withGrad {
		for i := range gp.dK {
			gp.dK[i] = mat.NewSymDense(len(gp.X), nil)
			gp.dK[i].Zero()
		}
	}

	if gp.Parallel {
		// Computing covariances in parallel --- for small
		// number of observations computing the covariance
		// matrix dominates the computation time.

		// argument buffer pools
		kpool := sync.Pool{
			New: func() interface{} {
				kargs := make([]float64, gp.Simil.NTheta()+2*gp.NDim)
				copy(kargs, gp.ThetaSimil)
				return kargs
			},
		}
		npool := sync.Pool{
			New: func() interface{} {
				nargs := make([]float64, gp.Noise.NTheta()+gp.NDim)
				copy(nargs, gp.ThetaNoise)
				return nargs
			},
		}

		// sync channel
		wait := make(chan bool, len(gp.X))

		// Ultimately, all elements of the covariance matrix
		// could be computed in parallel; however, the number of
		// goroutines would grow quadratically with the number
		// of inputs. A compromise taken here is to compute
		// all covariances in each row of the (upper triangle of)
		// the covariance matrix in parallel, spawning as many
		// goroutines as there are inputs.
		for i := range gp.X {
			go func(i int) {
				kargs := kpool.Get().([]float64)
				copy(kargs[gp.Simil.NTheta():], gp.X[i])
				for j := i; j != len(gp.X); j++ {
					nargs := npool.Get().([]float64)
					cov(i, j, kargs, nargs)
					npool.Put(nargs)
				}
				wait <- true
				kpool.Put(kargs)
			}(i)
		}

		// Wait for all goroutines to finish
		for range gp.X {
			<-wait
		}
	} else {
		kargs := make([]float64, gp.Simil.NTheta()+2*gp.NDim)
		nargs := make([]float64, gp.Noise.NTheta()+gp.NDim)
		copy(kargs, gp.ThetaSimil)
		copy(nargs, gp.ThetaNoise)

		for i := range gp.X {
			copy(kargs[gp.Simil.NTheta():], gp.X[i])
			for j := i; j != len(gp.X); j++ {
				cov(i, j, kargs, nargs)
			}
		}
	}

	if !gp.l.Factorize(K) {
		return fmt.Errorf("Factorize(%v)", mat.Formatted(K))
	}

	gp.alpha = mat.NewVecDense(len(gp.X), nil)
	err = gp.l.SolveVecTo(gp.alpha, mat.NewVecDense(len(gp.Y), gp.Y))
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
	for i := range x {
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

		if gp.Parallel {
			// Computing covariances in parallel --- for small
			// number of observations computing the covariance
			// matrix dominates the computation time.

			// argument buffer pool
			kpool := sync.Pool{
				New: func() interface{} {
					kargs := make([]float64, gp.Simil.NTheta()+2*gp.NDim)
					copy(kargs, gp.ThetaSimil)
					return kargs
				},
			}

			// sync channel
			wait := make(chan bool, len(x))

			for i := range gp.X {
				go func(i int) {
					kargs := kpool.Get().([]float64)
					copy(kargs[gp.Simil.NTheta():], gp.X[i])
					for j := range x {
						copy(kargs[gp.Simil.NTheta()+gp.NDim:], x[j])
						k := gp.Simil.Observe(kargs)
						model.DropGradient(gp.Simil)
						Kstar.Set(i, j, k)
					}
					wait <- true
					kpool.Put(kargs)
				}(i)
			}

			// Wait for all goroutines to finish
			for range gp.X {
				<-wait
			}
		} else {
			kargs := make([]float64, gp.Simil.NTheta()+2*gp.NDim)
			copy(kargs, gp.ThetaSimil)
			for i := range gp.X {
				copy(kargs[gp.Simil.NTheta():], gp.X[i])
				for j := range x {
					copy(kargs[gp.Simil.NTheta()+gp.NDim:], x[j])
					k := gp.Simil.Observe(kargs)
					model.DropGradient(gp.Simil)
					Kstar.Set(i, j, k)
				}
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
// model to infer the parameters.

// Observe computes log marginal likelihood of the parameters
// given the observations. The argument is the concatenation of
// log-transformed hyperparameters, inputs, and outputs.
//
// Optionally, the input can be only log-transformed
// hyperparameters, and then
// * only hyperparameters are inferred;
// * inputs must be assigned to fields X, Y of gp.
func (gp *GP) Observe(x []float64) float64 {
	gp.defaults()

	// Restore parameters from log scale
	theta := x[:gp.Simil.NTheta()+gp.Noise.NTheta()]
	for i := range theta {
		theta[i] = math.Exp(theta[i])
	}

	// Destructure
	copy(gp.ThetaSimil, model.Shift(&x, gp.Simil.NTheta()))
	copy(gp.ThetaNoise, model.Shift(&x, gp.Noise.NTheta()))
	gp.withObs = len(x) > 0
	if gp.withObs {
		// Observations are inferred as well as parameters,
		// normally as a part of a larger model with priors on
		// observations.
		n := len(x) / (gp.NDim + 1)
		gp.X = make([][]float64, n)
		for i := range gp.X {
			gp.X[i] = model.Shift(&x, gp.NDim)
		}
		gp.Y = model.Shift(&x, n)
	}
	if len(x) != 0 {
		panic("len(x)")
	}

	err := gp.absorb(withGradient)
	if err != nil {
		panic(err)
	}

	// Transform parameters back to log scale
	for i := range theta {
		theta[i] = math.Log(theta[i])
	}

	return gp.LML()
}

// Gradient computes the gradient of the log-likelihood with
// respect to the parameters and the inputs (GPML:5.9):
//   ∇L = ½ tr((α α^⊤ - Σ^−1) ∂Σ/∂θ), where α = Σ^-1 y
func (gp *GP) Gradient() []float64 {
	var grad []float64
	if gp.withObs {
		grad = make([]float64,
			gp.Simil.NTheta()+gp.Noise.NTheta()+len(gp.X)*(gp.NDim+1))
	} else {
		grad = make([]float64, gp.Simil.NTheta()+gp.Noise.NTheta())
	}

	if len(gp.X) == 0 {
		// no observations, return zero gradient
		return grad
	}

	// Gradient by parameters (and possibly inputs)
	// α α^⊤
	a := mat.NewDense(len(gp.Y), len(gp.Y), nil)
	a.Mul(gp.alpha, gp.alpha.T())
	if gp.Parallel {
		// register pool
		rpool := sync.Pool{
			New: func() interface{} {
				return mat.NewDense(len(gp.Y), len(gp.Y), nil)
			},
		}
		// sync channel
		wait := make(chan bool, len(gp.dK))

		for i := range gp.dK {
			go func(i int) {
				// registers
				// α α^⊤ ∂Σ/∂θ
				r0 := rpool.Get().(*mat.Dense)
				r0.Mul(a, gp.dK[i])
				// Σ^−1 ∂Σ/∂θ
				r1 := rpool.Get().(*mat.Dense)
				gp.l.SolveTo(r1, gp.dK[i])
				// (α α^⊤ - Σ^−1) ∂Σ/∂θ
				r2 := rpool.Get().(*mat.Dense)
				r2.Sub(r0, r1)
				rpool.Put(r0)
				rpool.Put(r1)

				grad[i] = 0.5 * mat.Trace(r2)
				wait <- true
				rpool.Put(r2)
			}(i)
		}

		// Wait for all goroutines to finish
		for range gp.dK {
			<-wait
		}
	} else {
		// registers
		r0 := mat.NewDense(len(gp.Y), len(gp.Y), nil)
		r1 := mat.NewDense(len(gp.Y), len(gp.Y), nil)
		r2 := mat.NewDense(len(gp.Y), len(gp.Y), nil)
		for i := range gp.dK {
			// α α^⊤ ∂Σ/∂θ
			r0.Mul(a, gp.dK[i])
			// Σ^−1 ∂Σ/∂θ
			gp.l.SolveTo(r1, gp.dK[i])
			// (α α^⊤ - Σ^−1) ∂Σ/∂θ
			r2.Sub(r0, r1)

			grad[i] = 0.5 * mat.Trace(r2)
		}
	}

	if gp.withObs {
		// Gradient by outputs
		for i := range gp.Y {
			grad[len(gp.dK)+i] = -gp.alpha.AtVec(i)
		}
	}

	// forget dK to release memory
	gp.dK = nil

	return grad
}
