package main

import (
	"bitbucket.org/dtolpin/gogp/gp"
	"bitbucket.org/dtolpin/gogp/tutorial"
	. "bitbucket.org/dtolpin/gogp/tutorial/warpedtime/kernel/ad"
	. "bitbucket.org/dtolpin/gogp/tutorial/warpedtime/model/ad"
	"bitbucket.org/dtolpin/infergo/model"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"strings"
)

var (
	LOGSIGMA float64
	SHOWWARP = false
)

func init() {
	LOGSIGMA = math.Log(0.5)
	flag.Usage = func() {
		fmt.Fprintf(flag.CommandLine.Output(),
			`A model with warped time. Invocation:
  %s [OPTIONS] < INPUT > OUTPUT
or
  %s [OPTIONS] selfcheck
In 'selfcheck' mode, the data hard-coded into the program is used,
to demonstrate basic functionality.
`, os.Args[0], os.Args[0])
		flag.PrintDefaults()
	}
	flag.Float64Var(&LOGSIGMA, "logsigma", LOGSIGMA,
		"log standard deviation of relative step")
	flag.BoolVar(&SHOWWARP, "show-warp", SHOWWARP,
		"show warped inputs")
}

type Model struct {
	gp           *gp.GP
	priors       *Priors
	gGrad, pGrad []float64
}

func (m *Model) Observe(x []float64) float64 {
	var gll, pll float64
	gll, m.gGrad = m.gp.Observe(x), model.Gradient(m.gp)
	pll, m.pGrad = m.priors.Observe(x), model.Gradient(m.priors)
	return gll + pll
}

func (m *Model) Gradient() []float64 {
	for i := range m.pGrad {
		m.gGrad[i] += m.pGrad[i]
	}

	// Wipe gradients of the first, last input and all outputs
	ixfirst := m.gp.Simil.NTheta() + m.gp.Noise.NTheta()
	ixlast := ixfirst + len(m.gp.X) - 1
	m.gGrad[ixfirst] = 0
	for i := ixlast; i != len(m.gGrad); i++ {
		m.gGrad[i] = 0
	}

	return m.gGrad
}

func main() {
	tutorial.OPTINP = true

	var (
		input  io.Reader = os.Stdin
		output io.Writer = os.Stdout
	)

	flag.Parse()
	switch {
	case flag.NArg() == 0:
	case flag.NArg() == 1 && flag.Arg(0) == "selfcheck":
		input = strings.NewReader(selfCheckData)
	default:
		panic("usage")
	}

	gp := &gp.GP{
		NDim:  1,
		Simil: Simil,
		Noise: Noise,
	}
	m := &Model{
		gp:     gp,
		priors: &Priors{LogSigma: LOGSIGMA},
	}
	theta := make([]float64, gp.Simil.NTheta()+gp.Noise.NTheta())

	// Collect results in a buffer to patch with updated inputs

	if SHOWWARP {
		buffer := strings.Builder{}
		tutorial.Evaluate(gp, m, theta, input, &buffer)
		// Predict at updated inputs
		mu, sigma, _ := gp.Produce(gp.X)

		// Patch
		lines := strings.Split(buffer.String(), "\n")
		ilast := len(lines) - 1
		if len(lines[ilast]) == 0 {
			// There is an extra empty line
			ilast--
		}
		for i, line := range lines[:ilast] {
			if len(line) == 0 {
				break
			}
			fields := strings.SplitN(line, ",", 5)
			fmt.Fprintf(output, "%f,%f,%f,%f,%s\n",
				gp.X[i][0], gp.Y[i], mu[i], sigma[i],
				fields[len(fields)-1])
		}

		// The last input is fixed, and the last line is left
		// unmodified
		fmt.Fprintln(output, lines[ilast])
	} else {
		tutorial.Evaluate(gp, m, theta, input, output)
	}
}

var selfCheckData = `0.1,-3.376024003717768007e+00
0.3,-1.977828720240523142e+00
0.5,-1.170229755402199645e+00
0.7,-9.583612412106726763e-01
0.9,-8.570477029219900622e-01
1.1,-8.907618364403485645e-01
1.3,-2.611461145416017482e-01
1.5,1.495844460881872728e-01
1.7,-4.165391766465373347e-01
1.9,-2.875013255153459069e-01
2.1,3.869524825854843142e-01
2.3,9.258652056784907325e-01
2.5,5.858145290237386504e-01
2.7,8.788023289396607041e-01
2.9,1.233057437482850682e+00
3.1,1.066540422694190138e+00
3.3,9.137144265931921305e-01
3.5,7.412075911286820640e-01
3.7,1.332146185234786673e+00
3.9,1.439962957400109378e+00
4.1,1.222960311200699257e+00
4.3,2.026371435028667956e-01
4.5,-1.659683673486037625e+00
4.7,-9.881392068563286113e-01
4.9,-3.948046844798779875e-01
5.1,-2.635420428119399916e-01
5.3,-1.610738281677652317e+00
5.5,-3.092358176820052540e-01
5.7,-2.958870744615414994e-01
5.9,-1.619124030623840138e+00
6.1,-1.241765328045226102e+00
6.3,-2.933200084576037536e-01
6.5,-6.066731986714126723e-01
6.7,5.866702176917204525e-01
6.9,6.282566869554838673e-01
7.1,1.013316587545910918e+00
7.3,1.123871563448763267e+00
7.5,1.094949286471081251e+00
7.7,1.113603299433020055e+00
7.9,8.567255613058102348e-01
8.1,7.384693873911447604e-01
8.3,3.434834982521656199e-01
8.5,-2.514717991306942083e-02
`
