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
				NTheta:      1,
				NDim:        1,
				Kernel:      kernel.Normal,
				NoiseKernel: kernel.ConstantNoise(0),
				Theta:       []float64{1.},
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
				NTheta:      1,
				NDim:        1,
				Kernel:      kernel.Normal,
				NoiseKernel: kernel.ConstantNoise(0),
				Theta:       []float64{1.},
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
				NTheta:      1,
				NDim:        1,
				Kernel:      kernel.Normal,
				NoiseKernel: kernel.ConstantNoise(0),
				Theta:       []float64{1.},
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
				NTheta:      1,
				NDim:        1,
				Kernel:      kernel.Normal,
				NoiseKernel: kernel.ConstantNoise(0),
				Theta:       []float64{1.},
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
				NTheta:      1,
				NDim:        1,
				Kernel:      kernel.Normal,
				NoiseKernel: kernel.ConstantNoise(0),
				Theta:       []float64{1.},
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
				NTheta:      1,
				NDim:        1,
				Kernel:      kernel.Normal,
				NoiseKernel: kernel.ConstantNoise(0.1),
				Theta:       []float64{1.},
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
			if math.Abs(mu[i]-c.mu[i]) > 1E-6 {
				t.Errorf("%s: wrong mu: got %v, want %v",
					c.name, mu, c.mu)
				break
			}
		}
		for i := range sigma {
			if math.Abs(sigma[i]-c.sigma[i]) > 1E-6 {
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
				NTheta:      1,
				NDim:        1,
				Kernel:      kernel.Normal,
				NoiseKernel: kernel.ConstantNoise(0),
			},
			x:   []float64{1.},
			ll:  0.0,
			dll: []float64{},
		},
		{
			name: "extra",
			gp: &GP{
				NTheta:      1,
				NDim:        1,
				Kernel:      kernel.Normal,
				NoiseKernel: kernel.ConstantNoise(0),
			},
			x:   []float64{1, 0, 1, 1, -1},
			ll:  0.0,
			dll: []float64{},
		},
	} {
		ll := c.gp.Observe(c.x)
		dll := c.gp.Gradient()
		ll = ll
		dll = dll
	}
}
