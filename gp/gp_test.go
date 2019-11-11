package gp

import (
	"bitbucket.org/dtolpin/gogp/kernel/ad"
	"math"
	"testing"
)

func TestProduce(t *testing.T) {
	for _, c := range []struct {
		name      string
		gp        *GP
		x         [][]float64
		y         []float64
		z         [][]float64
		mu, sigma []float64
	}{
		{
			name: "prior",
			gp: &GP{
				NDim:       1,
				Simil:      kernel.Normal,
				Noise:      kernel.ConstantNoise(0),
				ThetaSimil: []float64{1.},
			},
			x:     [][]float64{},
			y:     []float64{},
			z:     [][]float64{{0}},
			mu:    []float64{0},
			sigma: []float64{1},
		},
		{
			name: "1 self",
			gp: &GP{
				NDim:       1,
				Simil:      kernel.Normal,
				Noise:      kernel.ConstantNoise(0),
				ThetaSimil: []float64{1.},
			},
			x:     [][]float64{{0}},
			y:     []float64{1},
			z:     [][]float64{{0}},
			mu:    []float64{1},
			sigma: []float64{0},
		},
		{
			name: "two selves",
			gp: &GP{
				NDim:       1,
				Simil:      kernel.Normal,
				Noise:      kernel.ConstantNoise(0),
				ThetaSimil: []float64{1.},
			},
			x:     [][]float64{{0}, {1}},
			y:     []float64{1, -1},
			z:     [][]float64{{0}, {1}},
			mu:    []float64{1, -1},
			sigma: []float64{0, 0},
		},
		{
			name: "inter",
			gp: &GP{
				NDim:       1,
				Simil:      kernel.Normal,
				Noise:      kernel.ConstantNoise(0),
				ThetaSimil: []float64{1.},
			},
			x:     [][]float64{{0}, {1}},
			y:     []float64{1, -1},
			z:     [][]float64{{0.5}},
			mu:    []float64{0.},
			sigma: []float64{0.174518},
		},
		{
			name: "extra",
			gp: &GP{
				NDim:       1,
				Simil:      kernel.Normal,
				Noise:      kernel.ConstantNoise(0),
				ThetaSimil: []float64{1.},
			},
			x:     [][]float64{{0}, {1}},
			y:     []float64{1, -1},
			z:     [][]float64{{-2.}, {3.}},
			mu:    []float64{0.315720, -0.315720},
			sigma: []float64{0.986770, 0.986770},
		},
		{
			name: "noise",
			gp: &GP{
				NDim:       1,
				Simil:      kernel.Normal,
				Noise:      kernel.ConstantNoise(0.1),
				ThetaSimil: []float64{1.},
			},
			x:     [][]float64{{0}, {1}},
			y:     []float64{1, -1},
			z:     [][]float64{{-2.}, {3.}},
			mu:    []float64{0.251741, -0.251741},
			sigma: []float64{0.988979, 0.988979},
		},
	} {
		err := c.gp.Absorb(c.x, c.y)
		if err != nil {
			t.Fatalf("%s: absorb: %v", c.name, err)
		}
		mu, sigma, err := c.gp.Produce(c.z)
		if err != nil {
			t.Fatalf("%s: produce: %v", c.name, err)
		}
		if len(mu) != len(c.mu) {
			t.Errorf("%s: wrong len(mu): got %d, want %d",
				c.name, len(mu), len(c.mu))
		}
		if len(sigma) != len(c.sigma) {
			t.Errorf("%s: wrong len(sigma): got %d, want %d",
				c.name, len(sigma), len(c.sigma))
		}
		for i := range mu {
			if math.Abs(mu[i]-c.mu[i]) > 1e-6 {
				t.Errorf("%s: wrong mu: got %v, want %v",
					c.name, mu, c.mu)
				break
			}
		}
		for i := range sigma {
			if math.Abs(sigma[i]-c.sigma[i]) > 1e-6 {
				t.Errorf("%s: wrong sigma: got %v, want %v",
					c.name, sigma, c.sigma)
				break
			}
		}
	}
}

func TestElementalModel(t *testing.T) {
	for _, c := range []struct {
		name string
		gp   *GP
		x    []float64
		ll   float64
		dll  []float64
	}{
		{
			name: "prior",
			gp: &GP{
				NDim:  1,
				Simil: kernel.Normal,
				Noise: kernel.ConstantNoise(0),
			},
			x:   []float64{0},
			ll:  0,
			dll: []float64{0},
		},
		{
			name: "single",
			gp: &GP{
				NDim:  1,
				Simil: kernel.Normal,
				Noise: kernel.ConstantNoise(0),
			},
			x:   []float64{0, 0, 1},
			ll:  -1.418939,
			dll: []float64{-0, 0, -1},
		},
		{
			name: "nonoise",
			gp: &GP{
				NDim:  1,
				Simil: kernel.Normal,
				Noise: kernel.ConstantNoise(0),
			},
			x:  []float64{0, 0, 1, 1, 0},
			ll: -2.399528,
			dll: []float64{-0.338697,
				-0.338697, 0.338697,
				-1.581977, 0.959517},
		},
		{
			name: "withnoise",
			gp: &GP{
				NDim:  1,
				Simil: kernel.Normal,
				Noise: kernel.ConstantNoise(0.1),
			},
			x:  []float64{0, -2, -1, 1, 0},
			ll: -2.405074,
			dll: []float64{-0.133775,
				-0.133775, 0.133775,
				-1.306226, 0.720242},
		},
	} {
		ll := c.gp.Observe(c.x)
		dll := c.gp.Gradient()
		if math.Abs(ll-c.ll) >= 1e-6 {
			t.Errorf("%s: wrong log-likelihood: got %f, want %f",
				c.name, ll, c.ll)
		}
		if len(dll) != len(c.dll) {
			t.Errorf("%s: wrong gradient size: got %d, want %d",
				c.name, len(dll), len(c.dll))
			continue
		}
		for i := range dll {
			if math.Abs(dll[i]-c.dll[i]) >= 1e-6 {
				t.Errorf("%s: wrong gradient: got %v, want %v",
					c.name, dll, c.dll)
				break
			}
		}

		// test Observe with hyperparameters only
		x := c.x[:c.gp.Simil.NTheta()+c.gp.Noise.NTheta()]
		ll = c.gp.Observe(x)
		dll = c.gp.Gradient()
		if math.Abs(ll-c.ll) >= 1e-6 {
			t.Errorf("%s: wrong log-likelihood (hyperparameters only):"+
				" got %f, want %f", c.name, ll, c.ll)
		}
		if len(dll) != c.gp.Simil.NTheta()+c.gp.Noise.NTheta() {
			t.Errorf("%s: wrong gradient size (hyperparameters only):"+
				" got %d, want %d",
				c.name, len(dll), c.gp.Simil.NTheta()+c.gp.Noise.NTheta())
			continue
		}
		for i := range dll {
			if math.Abs(dll[i]-c.dll[i]) >= 1e-6 {
				t.Errorf("%s: wrong gradient (hyperparameters only):"+
					" got %v, want %v",
					c.name, dll,
					c.dll[:c.gp.Simil.NTheta()+c.gp.Noise.NTheta()])
				break
			}
		}
	}
}
