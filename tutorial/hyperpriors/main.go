package main

import (
	"bitbucket.org/dtolpin/gogp/gp"
	"bitbucket.org/dtolpin/gogp/tutorial"
	. "bitbucket.org/dtolpin/gogp/tutorial/hyperpriors/kernel/ad"
	. "bitbucket.org/dtolpin/gogp/tutorial/hyperpriors/model/ad"
	"bitbucket.org/dtolpin/infergo/ad"
	"bitbucket.org/dtolpin/infergo/model"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"
)

var (
	PARALLEL = false
)

func init() {
	flag.Usage = func() {
		fmt.Fprintf(flag.CommandLine.Output(),
			`A model with priors on hyperparameters. Invocation:
  %s < INPUT > OUTPUT
or
  %s selfcheck
In 'selfcheck' mode, the data hard-coded into the program is used,
to demonstrate basic functionality.
`, os.Args[0], os.Args[0])
		flag.PrintDefaults()
	}
	flag.BoolVar(&PARALLEL, "p", PARALLEL, "compute covariance in parallel")
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

	return m.gGrad
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

	if PARALLEL {
		ad.MTSafeOn()
	}

	gp := &gp.GP{
		NDim:     1,
		Simil:    Simil,
		Noise:    Noise,
		Parallel: ad.IsMTSafe(),
	}
	m := &Model{
		gp:     gp,
		priors: &Priors{},
	}
	theta := make([]float64, gp.Simil.NTheta()+gp.Noise.NTheta())
	tutorial.Evaluate(gp, m, theta, input, output)
}

var selfCheckData = `0.0,0.9175039317065515
0.39269908169872414,0.9731455919322247
0.7853981633974483,1.0959912384092567
1.1780972450961724,0.9736852086897476
1.5707963267948966,0.4775380771673724
1.9634954084936207,0.02846556338380002
2.356194490192345,0.06286810744669546
2.748893571891069,-0.3016264044260728
3.141592653589793,-0.233321219861381
3.5342917352885173,0.012264267282034447
3.9269908169872414,0.2754343006803267
4.319689898685965,0.5110954352674066
4.71238898038469,0.8800868378745963
5.105088062083414,1.2712889389325506
5.497787143782138,1.3786571999693404
5.890486225480862,2.0165791291805806
6.283185307179586,2.0609131663353013
6.675884388878311,1.6955708061671337
7.0685834705770345,1.593286155786137
7.461282552275758,1.4749136125592655
7.853981633974483,1.0919987686853532
8.246680715673207,0.8750887594366861
8.63937979737193,0.8246400547458113
9.032078879070655,-0.4606065733628699
9.42477796076938,0.4903858672596608
9.817477042468104,0.0602557212556416
10.210176124166829,0.3378772910180289
10.602875205865551,0.7467182768906508
10.995574287564276,1.2495571759539486
11.388273369263,1.5856961011065027
11.780972450961723,1.7902892401417951
12.173671532660448,2.00014671799282
12.566370614359172,2.7897051422694705
12.959069696057897,2.581713892655121
13.351768777756622,1.816659827097302
13.744467859455344,1.6346972278282976
14.137166941154069,1.3598566123665519
14.529866022852794,0.9607420336378067
14.922565104551516,0.5241283791944207
15.315264186250241,0.5008020492901891
15.707963267948966,0.1392328627433248
16.10066234964769,0.6519801688524172
16.493361431346415,0.9005734811102598
16.886060513045138,1.1072086180871397
`
