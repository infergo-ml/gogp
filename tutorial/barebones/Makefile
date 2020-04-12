all: selfcheck

selfcheck: barebones
	./barebones selfcheck

barebones: kernel/ad/kernel.go main.go ../tutorial.go ../../gp/gp.go
	go build .

kernel/ad/kernel.go: kernel/kernel.go
	deriv kernel

clean:
	rm -f ./barebones kernel/ad/*.go
