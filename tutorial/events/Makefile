all: selfcheck

selfcheck: events
	./events -events 1.0:1.0:0.5,4.2:6.7:0.25 selfcheck

events: kernel/ad/kernel.go main.go ../tutorial.go
	go build .

kernel/ad/kernel.go: kernel/kernel.go
	deriv kernel

clean:
	rm -f ./events kernel/ad/*.go
