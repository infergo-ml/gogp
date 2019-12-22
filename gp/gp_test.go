package gp

import (
	"bitbucket.org/dtolpin/gogp/kernel/ad"
	"bitbucket.org/dtolpin/infergo/ad"
	"math"
	"testing"
)

func init() {
	ad.MTSafeOn()
}

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
			name: "self",
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
			name: "next",
			gp: &GP{
				NDim:       1,
				Simil:      kernel.Normal,
				Noise:      kernel.ConstantNoise(0),
				ThetaSimil: []float64{1.},
			},
			x:     [][]float64{{0}},
			y:     []float64{0},
			z:     [][]float64{{1}},
			mu:    []float64{0},
			sigma: []float64{0.795060},
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
			mu:    []float64{0.307895, -0.307895},
			sigma: []float64{0.987037, 0.987037},
		},
	} {
		warnedParallel := false
		for _, parallel := range []bool{false, true} {
			if parallel && !ad.IsMTSafe() {
				if !warnedParallel {
					t.Log("parallel computation is not supported")
					warnedParallel = true
					continue
				}

			}
			c.gp.Parallel = parallel
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
}

// Difference and precision for numerical derivative
const (
	dx  = 1e-8
	eps = 1e-4
)

func TestElementalModel(t *testing.T) {
	for _, c := range []struct {
		name string
		gp   *GP
		x    []float64
		ll   float64
	}{
		{
			name: "prior",
			gp: &GP{
				NDim:  1,
				Simil: kernel.Normal,
				Noise: kernel.ConstantNoise(0),
			},
			x:  []float64{0},
			ll: 0,
		},
		{
			name: "single",
			gp: &GP{
				NDim:  1,
				Simil: kernel.Normal,
				Noise: kernel.ConstantNoise(0),
			},
			x:  []float64{0, 0, 1},
			ll: -1.418939,
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
		},
		{
			name: "withnoise",
			gp: &GP{
				NDim:  1,
				Simil: kernel.Normal,
				Noise: kernel.ConstantNoise(0.1),
			},
			x:  []float64{1, -2, -1, 1, 0},
			ll: -4.321055,
		},
		{
			name: "uninoise",
			gp: &GP{
				NDim:  1,
				Simil: kernel.Normal,
				Noise: kernel.UniformNoise,
			},
			x:  []float64{1, 1, -1, -1, 1, 0},
			ll: -4.018110,
		},
	} {
		ll := c.gp.Observe(c.x)
		dll := c.gp.Gradient()
		if math.Abs(ll-c.ll) >= 1e-6 {
			t.Errorf("%s: wrong log-likelihood: got %f, want %f",
				c.name, ll, c.ll)
		}
		if len(dll) != len(c.x) {
			t.Errorf("%s: wrong gradient size: got %d, want %d",
				c.name, len(dll), len(c.x))
			continue
		}
		for j := range c.x {
			x0 := c.x[j]
			c.x[j] += dx
			llj := c.gp.Observe(c.x)
			dldx := (llj - ll) / dx
			c.x[j] = x0
			if math.Abs(dll[j]-dldx) > eps {
				t.Errorf("%s: dl/dx%d mismatch: got %.4f, want %.4f",
					c.name, j, dldx, dll[j])
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
	}
}
