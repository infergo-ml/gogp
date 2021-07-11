package gp

import (
	"bitbucket.org/dtolpin/infergo/model"
)

// Type Model is the wrapper model combining a GP instance and
// priors on the hyperparameters.
type Model struct {
	*GP                      // GP instance
	Priors       model.Model // hyperparameter priors
	gGrad, pGrad []float64   // GP and priors gradients
}

func (m *Model) Observe(x []float64) float64 {
	var gll, pll float64
	gll, m.gGrad = m.GP.Observe(x), model.Gradient(m.GP)
	pll, m.pGrad = m.Priors.Observe(x), model.Gradient(m.Priors)
	return gll + pll
}

func (m *Model) Gradient() []float64 {
	for i := range m.pGrad {
		m.gGrad[i] += m.pGrad[i]
	}

	return m.gGrad
}
