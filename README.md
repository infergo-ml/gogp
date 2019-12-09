# Gaussian Process regression in Go

GoGP is a library for probabilistic programming around Gaussian Processes,
in Go. A Gaussian process is by itself an [Infergo](http://infergo.org)
model (elemental, with custom gradient). A GP model can be used within
a larger model, for example to handle uncertain inputs, non-Gaussian
noise, or impose priors on GP hyperparameters.

Kernels are automatically differentiated by leveraging the reverse-mode
automatic differentation of Infergo.

# Examples

More examples in the [tutorial](/dtolpin/gogp/src/master/tutoral/) folder.

## Bare bones

In the basic case, similar to that supported by many Gaussian
process libraries, a GP directly serves as the model for
inference on hyperparameters (or the hyperparameters can be just
fixed).

The library user specifies the kernel:
```Go
type Basic struct{}
func (Basic) Observe(x []float64) float64 {
    return x[0] * kernel.Normal.Observe(x[1:])
}
func (Basic) NTheta() int { return 2 }
```
and initializes `GP` with a kernel instance:
```Go
gp := &gp.GP{
    NDim:  1,
    Simil: Basic{},
}
\end{lstlisting}
```

MLE inference on hyperparameters and prediction can then be performed
through library functions.

## Priors on hyperparameters

If priors on hyperparameters are to be specified, the library
user provides both the kernel and the model.  `GP` is
initialized with the kernel, and then `GP` and the
model are combined for inference in a derived model:
```Go
type Model struct {
    gp           *gp.GP
    priors       *Priors
    gGrad, pGrad []float64
}
func (m *Model) Observe(x []float64) float64 {
    var gll, pll float64
    gll, m.gGrad = \
      m.gp.Observe(x), model.Gradient(m.gp)
    pll, m.pGrad = \
      m.priors.Observe(x),model.Gradient(m.priors)
    return gll + pll
}
func (m *Model) Gradient() []float64 {
	for i := range m.pGrad {
		m.gGrad[i] += m.pGrad[i]
	}
	return m.gGrad
}
```
In `Model`, `gp` holds a GP
instance and `priors` holds an instance of the model
expressing beliefs about hyperparameters. A `Model`
instance is used for inference on hyperparameters, a
`GP` instance --- for prediction.
