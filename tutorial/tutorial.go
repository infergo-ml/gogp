package tutorial

import (
	"bitbucket.org/dtolpin/gogp/gp"
	"encoding/csv"
	"io"
)

// Evaluate evaluates Gaussian process gp on the CSV data
// read from reader r.
func Evaluate(gp *gp.GP, rdr io.Reader, wtr io.Writer) error {
	// Load the data
	x, y, err := load(rdr)
	if err != nil {
		return err
	}
	// Forecast one step out of sample, iteratively.
	x = x
	y = y
	// Output data augmented with predictions.
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
		case io.EOF:
			// end of file
			break
		default:
			return x, y, err
		}
		record = record
	}
	return x, y, err
}
