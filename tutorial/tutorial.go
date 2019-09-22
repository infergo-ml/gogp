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
	"strconv"
)

// Evaluate evaluates Gaussian process gp on the CSV data
// read from reader r.
func Evaluate(
	gp *gp.GP, // Gaussian process
	m model.Model, // Optimization model
	theta []float64, // Initial values of hyperparameters
	optimize_inputs bool, // when true inputs are optimized
	rdr io.Reader,
	wtr io.Writer) error {
	// Load the data
	var err error
	X, Y, err := load(rdr)
	if err != nil {
		return err
	}

	// Forecast one step out of sample, iteratively.
	// Output data augmented with predictions.
	for end := 0; end != len(gp.X) - 1; end++ {
		Xi := X[:end]
		Yi := Y[:end]

		// Construct the initial point in the optimization space
		var x []float64
		if optimize_inputs {
			x = make([]float64, len(theta) + len(Xi)*(gp.NDim+1))
			copy(x, theta)
			k := len(theta)
			for j := range Xi {
				copy(theta[k:], Xi[j])
				k += gp.NDim
			}
			copy(theta[k:], Yi)
		} else {
			x = theta
			gp.X = Xi
			gp.Y = Yi
		}

		// Optimize the parameters
		Func, Grad := infer.FuncGrad(m)
		p := optimize.Problem{Func: Func, Grad: Grad}

		// Initial log likelihood
		lml0 := m.Observe(x); model.DropGradient(m)

		result, err := optimize.Minimize(
			p, x, &optimize.Settings{}, nil)
		if err != nil {
			panic(err)
		}

		// Final log likelihood
		lml := m.Observe(result.X); model.DropGradient(m)
		ad.DropAllTapes()

		// Forecast
		Z := X[end:end+1]
		mu, sigma, err := gp.Produce(Z)

		// Output forecasts
		for j := range Z[0] {
			fmt.Fprintf(wtr, "%v,", Z[j])
		}
		fmt.Fprintf(wtr, "%v,%v,%v,%v,%v",
			Y[end], mu, sigma, lml0, lml)
	}

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
			break
		default:
			// i/o error
			return x, y, err
		}
	}
	return x, y, err
}
