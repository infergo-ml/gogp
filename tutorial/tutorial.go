package tutorial

import (
	"bitbucket.org/dtolpin/gogp/gp"
	"bitbucket.org/dtolpin/infergo/ad"
	"bitbucket.org/dtolpin/infergo/infer"
	"bitbucket.org/dtolpin/infergo/model"
	"encoding/csv"
    "flag"
	"fmt"
	"gonum.org/v1/gonum/optimize"
	"gonum.org/v1/gonum/stat"
	"io"
	"math"
	"math/rand"
	"os"
	"strconv"
	"time"
)

var (
	OPTINP    = false
	MINOPT    = 0
	ALG       = "lbfgs"
	PARALLEL  = false
	ITERS     = 1000 // major iterations
	MINITERS  = 10   // minimum iterations to accept in lbfgs
	THRESHOLD = 1e-6 // gradient threshold
	RATE      = 0.01 // learning rate (for Adam)
	NTASKS    = 0
    NONORMALIZE = false
    OUTOFSAMPLE = false
)

func init() {
	rand.Seed(time.Now().UnixNano())
	flag.StringVar(&ALG, "a", ALG,
		"optimization algorithm + adam or lbfgs)")
	flag.BoolVar(&PARALLEL, "p", PARALLEL,
		"compute covariance in parallel")
	flag.BoolVar(&NONORMALIZE, "n", NONORMALIZE,
		"normalize outputs")
	flag.BoolVar(&OUTOFSAMPLE, "o", OUTOFSAMPLE,
		"forecast out of sample")
}

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
	if PARALLEL {
		ad.MTSafeOn()
	}
	gp.Parallel = ad.IsMTSafe()

	// Load the data
	var err error
	fmt.Fprint(os.Stderr, "loading...")
	X, Y, err := load(rdr)
	if err != nil {
		return err
	}
	fmt.Fprintln(os.Stderr, "done")

	// Normalize Y
    var meany, stdy float64
    if NONORMALIZE {
        meany, stdy = 0., 1.
    } else {
        meany, stdy = stat.MeanStdDev(Y, nil)
        for i := range Y {
            Y[i] = (Y[i] - meany) / stdy
        }
    }

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
			x = make([]float64, len(theta))
			copy(x, theta)
			gp.X = Xi
			gp.Y = Yi
		}

		// Randomize the initial values of hyperparameters
		for i := range theta {
			x[i] += 0.1 * rand.NormFloat64()
		}

		// Initial log likelihood
		lml0 := m.Observe(x)
		model.DropGradient(m)

		if len(gp.X) > MINOPT {
			switch ALG {
			case "lbfgs":
				// Optimize the parameters
				Func, Grad := infer.FuncGrad(m)
				p := optimize.Problem{Func: Func, Grad: Grad}

				// For some kernels and data, the optimizing of
				// hyperparameters does not make sense with too few
				// points.
				result, err := optimize.Minimize(
					p, x, &optimize.Settings{
						MajorIterations:   ITERS,
						GradientThreshold: THRESHOLD,
						Concurrent:        NTASKS,
					}, nil)
				// We do not need the optimizer to `officially'
				// converge, a few iterations usually bring most
				// of the improvement. However, in pathological
				// cases even a few iterations do not succeed,
				// and we want to report that.
				if err != nil && result.Stats.MajorIterations <= MINITERS {
					// There was a problem and the optimizer stopped
					// too early.
					fmt.Fprintf(os.Stderr,
						"%d: stuck after %d iterations: %v\n",
						end, result.Stats.MajorIterations, err)
				}
				x = result.X
			case "adam":
				opt := &infer.Adam{Rate: RATE}
				epoch := 0
			Epochs:
				for ; epoch != ITERS; epoch++ {
					_, grad := opt.Step(m, x)
					for i := range grad {
						if math.Abs(grad[i]) >= THRESHOLD {
							continue Epochs
						}
					}
					break Epochs
				}
			}
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
			Y[end]*stdy + meany, 
            mu[0]*stdy + meany,
            sigma[0]*stdy,
            lml0, lml)
		for i := 0; i != len(theta); i++ {
			fmt.Fprintf(wtr, ",%f", math.Exp(x[i]))
		}
		fmt.Fprintln(wtr)
	}

    if OUTOFSAMPLE {
        Z := make([][]float64,  len(X))
        for i := range Z {
            Z[i] = make([]float64, len(X[0]))
            copy(Z[i], X[i])
            for j := range Z[i] {
                Z[i][j] += X[len(X)-1][j]
            }
        }
        Z = Z[1:]

        mu, sigma, err := gp.Produce(Z)
        if err != nil {
            fmt.Fprintf(os.Stderr, "Failed to forecast: %v\n", err)
        }

        // Output forecasts
        for i := range Z {
            z := Z[i]
            for j := range z {
                fmt.Fprintf(wtr, "%f,", z[j])
            }
            fmt.Fprintf(wtr, "nan,%f,%f", mu[i]*stdy + meany, sigma[i]*stdy)
            fmt.Fprintln(wtr)
        }
    }

	fmt.Fprintln(os.Stderr, "done")

	return nil
}

// load parses the data from csv and returns inputs and outputs,
// suitable for feeding to the GP.
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
