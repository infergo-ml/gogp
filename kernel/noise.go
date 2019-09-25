package kernel

// Noise kernels
// 
// A noise kernel is used to add noise to diagonal elements
// of the covariance matrix. It is different from a similarity
// kernel in that it accepts a single input location.
//
// Noise kernels are common in implementations of GP. However,
// when the GP is a part of a larger model, a more flexible and
// consistent way to account for noise is to treat GP inputs as
// latent variables with conditional priors. For example,
// normal noise can be represented as a normal prior on the
// latent input centered around the observed input. Noise
// variance can be specified or inferred in the model
// incrorporating the GP.

// ConstantNoise is a noise kernel assigning the same fixed
// noise to all points. Used as a default when no noise kernel
// is given.
type ConstantNoise float64

func (nk ConstantNoise) Observe(x []float64) float64 {
	return float64(nk)
}

func (ConstantNoise) NTheta() int {
	return 0
}

// UniformNoise is a noise kernel for learning the same noise
// variance for all points.
type uniformNoise struct{}

var UniformNoise uniformNoise

func (nk uniformNoise) Observe(x []float64) float64 {
	return x[0]
}

func (uniformNoise) NTheta() int {
	return 1
}
