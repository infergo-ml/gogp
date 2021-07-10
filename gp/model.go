package gp

import (
	"bitbucket.org/dtolpin/infergo/model"
)

type Model struct {
	*GP
	Priors       model.Model
	gGrad, pGrad []float64
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
