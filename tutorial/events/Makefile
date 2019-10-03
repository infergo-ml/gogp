all: selfcheck

selfcheck: events
	./events selfcheck

events: kernel/ad/kernel.go main.go ../tutorial.go
	go build .

kernel/ad/kernel.go: kernel/kernel.go
	deriv kernel

clean:
	rm -f ./events kernel/ad/*.go
