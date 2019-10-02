package tutorial

import (
	"bitbucket.org/dtolpin/gogp/gp"
	"bitbucket.org/dtolpin/infergo/ad"
	"bitbucket.org/dtolpin/infergo/infer"
	"bitbucket.org/dtolpin/infergo/model"
	"encoding/csv"
	"fmt"
	"gonum.org/v1/gonum/optimize"
	"io"
	"os"
	"strconv"
)

var (
	NITER          = 0
	EPS    float64 = 0
	NTASKS         = 0
	OPTINP         = false
	MINOPT         = 0
)

// Evaluate evaluates Gaussian process on CSV data.  One step
// out of sample forecast is recorded for each time point, along
// with the hyperparameters. This function is called by all
// case studies in the tutorial. For optimization, LBFGS from
// the gonum library (http://gonum.org) is used for faster
// execution. In general though, LBFGS is a bit of hit-or-miss,
// failing to optimize occasionally, so in real applications a
// different optimization/inference algorithm may be a better
// choice.
func Evaluate(
	gp *gp.GP, // gaussian process
	m model.Model, // optimization model
	theta []float64, // initial values of hyperparameters
	rdr io.Reader, // data
	wtr io.Writer, // forecasts
) error {
	// Load the data
	var err error
	fmt.Fprint(os.Stderr, "loading...")
	X, Y, err := load(rdr)
	if err != nil {
		return err
	}
	fmt.Fprintln(os.Stderr, "done")

	// Forecast one step out of sample, iteratively.
	// Output data augmented with predictions.
	fmt.Fprintln(os.Stderr, "Forecasting...")
	for end := 0; end != len(X); end++ {
		Xi := X[:end]
		Yi := Y[:end]

		// Construct the initial point in the optimization space
		var x []float64
		if OPTINP {
			// If the inputs are optimized as well as the
			// hyperparameters, the inputs are appended to the
			// parameter vector of Observe.
			x = make([]float64, len(theta)+len(Xi)*(gp.NDim+1))
			copy(x, theta)
			k := len(theta)
			for j := range Xi {
				copy(x[k:], Xi[j])
				k += gp.NDim
			}
			copy(x[k:], Yi)
		} else {
			// If only the hyperparameters are optimized, the
			// inputs are stored in the fields of the GP.
			x = theta
			gp.X = Xi
			gp.Y = Yi
		}

		// Optimize the parameters
		Func, Grad := infer.FuncGrad(m)
		p := optimize.Problem{Func: Func, Grad: Grad}

		// Initial log likelihood
		lml0 := m.Observe(x)
		model.DropGradient(m)

		// For some kernels and data, the optimizing of
		// hyperparameters does not make sense with too few
		// points.
		if len(gp.X) > MINOPT {
			result, err := optimize.Minimize(
				p, x, &optimize.Settings{
					MajorIterations:   NITER,
					GradientThreshold: EPS,
					Concurrent:        NTASKS,
				}, nil)
			// We do not need the optimizer to `officially'
			// converge, a few iterations usually bring most
			// of the improvement. However, in pathological
			// cases even a single iteration does not succeed,
			// and we want to report that.
			if err != nil && result.Stats.MajorIterations == 1 {
				// There was a problem and the optimizer stopped
				// on first iteration.
				fmt.Fprintf(os.Stderr, "Failed to optimize: %v\n", err)
			}
			x = result.X
		}

		// Final log likelihood
		lml := m.Observe(x)
		model.DropGradient(m)
		ad.DropAllTapes()

		// Forecast
		Z := X[end : end+1]
		mu, sigma, err := gp.Produce(Z)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Failed to forecast: %v\n", err)
		}

		// Output forecasts
		z := Z[0]
		for j := range z {
			fmt.Fprintf(wtr, "%f,", z[j])
		}
		fmt.Fprintf(wtr, "%f,%f,%f,%f,%f",
			Y[end], mu[0], sigma[0], lml0, lml)
		for i := range gp.ThetaSimil {
			fmt.Fprintf(wtr, ",%f", gp.ThetaSimil[i])
		}
		for i := range gp.ThetaNoise {
			fmt.Fprintf(wtr, ",%f", gp.ThetaNoise[i])
		}
		fmt.Fprintln(wtr)
	}
	fmt.Fprintln(os.Stderr, "done")

	return nil
}

// load parses the data from csv and returns input locations
// and inputs, suitable for feeding to the GP.
func load(rdr io.Reader) (
	x [][]float64,
	y []float64,
	err error,
) {
	csv := csv.NewReader(rdr)
RECORDS:
	for {
		record, err := csv.Read()
		switch err {
		case nil:
			// record contains the data
			xi := make([]float64, len(record)-1)
			i := 0
			for ; i != len(record)-1; i++ {
				xi[i], err = strconv.ParseFloat(record[i], 64)
				if err != nil {
					// data error
					return x, y, err
				}
			}
			yi, err := strconv.ParseFloat(record[i], 64)
			if err != nil {
				// data error
				return x, y, err
			}
			x = append(x, xi)
			y = append(y, yi)
		case io.EOF:
			// end of file
			break RECORDS
		default:
			// i/o error
			return x, y, err
		}
	}

	return x, y, err
}
