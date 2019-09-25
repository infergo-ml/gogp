all: selfcheck

selfcheck: hyperpriors
	./hyperpriors selfcheck

hyperpriors: kernel/ad/kernel.go main.go ../tutorial.go
	go build .

kernel/ad/kernel.go: kernel/kernel.go
	deriv kernel

clean:
	rm -f ./hyperpriors kernel/ad/*.go
