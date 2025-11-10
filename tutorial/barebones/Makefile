all: selfcheck

GO=go

selfcheck: barebones
	./barebones selfcheck
	./barebones -p selfcheck

barebones: kernel/ad/kernel.go main.go ../tutorial.go ../../gp/gp.go
	$(GO) build .

kernel/ad/kernel.go: kernel/kernel.go
	deriv kernel

clean:
	rm -f ./barebones kernel/ad/*.go
