package main

import (
	"bitbucket.org/dtolpin/gogp/gp"
	"bitbucket.org/dtolpin/gogp/kernel/ad"
	"bitbucket.org/dtolpin/gogp/tutorial"
	. "bitbucket.org/dtolpin/gogp/tutorial/barebones/kernel/ad"
	"io"
	"flag"
	"os"
	"strings"
)

func main() {
	var (
		input io.Reader = os.Stdin
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

	gp := &gp.GP {
		NTheta: 2,
		NNoiseTheta: 1,
		NDim: 1,
		Kernel: Kernel,
		NoiseKernel: kernel.ConstantNoise(0.0001),
	}
	theta := make([]float64, gp.NTheta + gp.NNoiseTheta)
	theta[2] = -10
	tutorial.Evaluate(gp, gp, theta, false, input, output)
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
