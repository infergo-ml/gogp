package gp

import (
	. "bitbucket/org/dtolpin/infergo/model"
)

type GP struct {
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
