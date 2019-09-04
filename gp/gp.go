package gp

import (
	. "bitbucket/org/dtolpin/infergo/model"
)

type GP struct {
	NPar, NDim int       // dimensions
	Kernel     Model     // kernel
	Theta      []float64 // kernel parameters
}

// Absorb absorbs observations into the process
func (gp GP) Absorb(x [][]float64, y []float64) {
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
