package gputil

import (
	. "bitbucket.org/dtolpin/gogp/gp"
	"bitbucket.org/dtolpin/infergo/model"
	"math"
)

// Log-scale model for MLE parameter estimation by gradient
// optimization.

type logscale struct {
	*GP
	x []float64
}

// trick for calling parent's methods on logscale
type linscale = struct {
	*GP
	x []float64
}

// LogScale returns a model with parameters mapped to the
// log-scale and no inference on input locations. The model
// is suitable for MLE estimation of GP parameters.
func LogScale(gp *GP, x [][]float64, y []float64) model.Model {
	ls := &logscale{
		gp,
		make([]float64,
			gp.NTheta+gp.NNoiseTheta+len(x)*(gp.NDim+1)),
	}
	xy := ls.x[gp.NTheta+gp.NNoiseTheta:]
	for i := 0; i != len(x); i++ {
		copy(xy, x[i])
		xy = xy[gp.NDim:]
	}
	copy(xy, y)
	return ls
}

func (ls *logscale) Observe(theta []float64) float64 {
	for i := range theta {
		ls.x[i] = math.Exp(theta[i])
	}
	ll := (*linscale)(ls).Observe(ls.x)
	return ll
}

func (ls *logscale) Gradient() []float64 {
	grad := (*linscale)(ls).Gradient()
	grad = grad[:ls.NTheta+ls.NNoiseTheta]
	i := 0
	for _, t := range ls.Theta {
		grad[i] *= t
		i++
	}
	for _, t := range ls.NoiseTheta {
		grad[i] *= t
		i++
	}
	return grad
}
