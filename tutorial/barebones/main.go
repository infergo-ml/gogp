package main

import (
	. "bitbucket.org/dtolpin/gogp/gp"
	. "bitbucket.org/dtolpin/gogp/tutorial"
	. "bitbucket.org/dtolpin/gogp/tutorial/barebones/kernel/ad"
	"bitbucket.org/dtolpin/infergo/ad"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"
)

func init() {
	flag.Usage = func() {
		fmt.Fprintf(flag.CommandLine.Output(),
			`A bare bones time series forecasting with gogp. RBF similarity
and uniform noise kernels are used. This basic use is similar to
GP regression with scikit-learn, for example. Invocation:
  %s [OPTIONS] < INPUT > OUTPUT
or
  %s selfcheck
In 'selfcheck' mode, the data hard-coded into the program is used,
to demonstrate basic functionality.
`, os.Args[0], os.Args[0])
		flag.PrintDefaults()
	}
}

func main() {
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

	gp := &GP{
		NDim:     1,
		Simil:    Simil,
		Noise:    Noise(0.01),
		Parallel: ad.IsMTSafe(),
		// 0.01 is the `prior', or rather the starting search
		// point for input noise; see kernel/kernel.go for
		// details. We might modify the initial point instead.
	}
	fmt.Println(ad.IsMTSafe())
	theta := make([]float64, gp.Simil.NTheta()+gp.Noise.NTheta())
	Evaluate(gp, gp, theta, input, output)
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
