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
// L = −½ log|Σ| − ½ (y−μ)^⊤ Σ^−1 (y−μ) − n/2 log(2π)
func (gp GP) ll() float64 {
	return 0.
}

// gradll computes the gradient of the log-likelihood with respect
// to the parameters and the input data locations
func (gp GP) gradll() []float64 {
	// ∇L = ½ tr(Σ^−1 ∂Σ/∂θ)􏰃+ ½ (y−μ)^⊤ ∂Σ/∂θ Σ^−1 ∂Σ/∂θ (y−μ)
	return []float64{}
}
