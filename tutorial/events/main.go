package main

import (
	. "bitbucket.org/dtolpin/gogp/gp"
	. "bitbucket.org/dtolpin/gogp/tutorial"
	. "bitbucket.org/dtolpin/gogp/tutorial/events/kernel/ad"
	"flag"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
)

var (
	EVENTS = ""
)

func init() {
	flag.Usage = func() {
		fmt.Fprintf(flag.CommandLine.Output(),
			`A kernel with stretched time around events. Invocation:
  %s [OPTIONS] < INPUT > OUTPUT
or
  %s [OPTIONS] selfcheck
In 'selfcheck' mode, the data hard-coded into the program is used,
to demonstrate basic functionality.
`, os.Args[0], os.Args[0])
		flag.PrintDefaults()
	}
	flag.StringVar(&EVENTS, "events", EVENTS,
		"comma separated colon connected event list \"from:to:discount,...\", "+
			"for example \"1.:2.5:0.3,3:6:0.5\"")
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

	simil := &Simil{}
	if len(EVENTS) > 0 {
		for _, event := range strings.Split(EVENTS, ",") {
			var e []float64
			for _, s := range strings.Split(event, ":") {
				v, err := strconv.ParseFloat(s, 64)
				if err != nil {
					panic(err)
				}
				e = append(e, v)
			}
			simil.Events = append(simil.Events, e)
		}
	}

	gp := &GP{
		NDim:  1,
		Simil: simil,
		Noise: Noise(0.01),
		// 0.01 is the `prior', or rather the starting search
		// point for input noise; see kernel/kernel.go for
		// details. We might modify the initial point instead.
	}

	theta := make([]float64, gp.Simil.NTheta()+gp.Noise.NTheta())
	Evaluate(gp, gp, theta, input, output)
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
