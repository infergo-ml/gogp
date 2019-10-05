package main

import (
	"bitbucket.org/dtolpin/gogp/gp"
	"bitbucket.org/dtolpin/gogp/tutorial"
	. "bitbucket.org/dtolpin/gogp/tutorial/anynoise/kernel/ad"
	. "bitbucket.org/dtolpin/gogp/tutorial/anynoise/model/ad"
	"bitbucket.org/dtolpin/infergo/model"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"
)

func init() {
	flag.Usage = func() {
		fmt.Fprintf(flag.CommandLine.Output(),
			`A model with non-gaussian noise. Invocation:
  %s [OPTIONS] < INPUT > OUTPUT
or
  %s [OPTIONS] selfcheck
In 'selfcheck' mode, the data hard-coded into the program is used,
to demonstrate basic functionality.
`, os.Args[0], os.Args[0])
		flag.PrintDefaults()
	}
}

type Model struct {
	gp           *gp.GP
	priors       *Priors
	gGrad, pGrad []float64
}

func (m *Model) Observe(x []float64) float64 {
	var gll, pll float64
	gll, m.gGrad = m.gp.Observe(x[1:]), model.Gradient(m.gp)
	pll, m.pGrad = m.priors.Observe(x), model.Gradient(m.priors)
	return gll + pll
}

func (m *Model) Gradient() []float64 {
	for i := range m.gGrad {
		m.pGrad[i+1] += m.gGrad[i]
	}

	// Wipe gradients of inputs but keep gradients of outputs
	ixfirst := 1 + m.gp.Simil.NTheta() + m.gp.Noise.NTheta()
	iyfirst := ixfirst + len(m.gp.X)
	for i := ixfirst; i != iyfirst; i++ {
		m.pGrad[i] = 0
	}

	return m.pGrad
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
	}
	m := &Model{
		gp:     gp,
		priors: &Priors{},
	}
	theta := make([]float64, 1+gp.Simil.NTheta())
	tutorial.Evaluate(gp, m, theta, input, output)
}

var selfCheckData = `0.0,-0.04322589452340684
0.3141592653589793,0.24791846152402336
0.6283185307179586,0.2802626294538516
0.9424777960769379,0.8412528753369979
1.2566370614359172,0.95401852884978
1.5707963267948966,0.9058860181087818
1.8849555921538759,0.9216334800816051
2.199114857512855,0.710326937917382
2.5132741228718345,0.698293927016971
2.827433388230814,0.5121958179707433
3.141592653589793,-0.018245933166371128
3.455751918948772,-0.23666711312329597
3.7699111843077517,-0.587279399315788
4.084070449666731,-0.7078602038425731
4.39822971502571,-0.8404888142915089
4.71238898038469,-0.9726473641825701
5.026548245743669,-1.0070152669699057
5.340707511102648,-0.7681868685956548
5.654866776461628,-0.773709471068197
5.969026041820607,-0.19551568791123064
`
