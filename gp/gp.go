package gp

import (
	. "bitbucket/org/dtolpin/infergo/model"
)

type GP struct {
	x      [][]float64
	y      []float64
	Kernel Model
	Theta  []float64
}

// Absorb absorbs observations into the process
func (gp GP) Absorb(x [][]float64, y []float64) {
}

// Produce computes predictions
func (gp GP) Produce(x [][]float64) (mu, sigma []float64) {
}

// Train optimizes hyperparameters given observations
func (gp GP) Train(x [][]float64, y []float64) {
}

// ll computes log-likelihood of the parameters given the data
// (GPML:5.8):
//   L = −½ log|Σ| − ½ y^⊤ Σ^−1 y − n/2 log(2π)

func (gp GP) ll() float64 {
	return 0.
}

// gradll computes the gradient of the log-likelihood with respect
// to the parameters and the input data locations (GPML:5.9):
//   ∇L = ½ tr((α α^⊤ - Σ^−1) ∂Σ/∂θ), where α = Σ^-1 y
func (gp GP) gradll() []float64 {
	return []float64{}
}
